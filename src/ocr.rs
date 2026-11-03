//! 纯 Rust 的中文文字识别，接口保持 `cnocr_rs::ocr::{Ocr, OcrTraitConst}` 可用。
//!
//! 实现参考 CnocrCpp：多模型注册表、按 token 行读取字典、依据会话元数据组装输入/输出、
//! 按模型决定通道数与归一化方式，以及基于反色投影的文字行切分。

use std::path::Path;
use std::sync::Mutex;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, TensorRef, ValueType};

use crate::model_registry::{dict_path, find_recognizer, model_path, ModelSpec, UseModel};

/// 单行图送入模型前的最小宽度
const MIN_LINE_WIDTH: usize = 8;
/// 反色后行内最大像素超过该阈值即认为该行有文字
const ROW_TEXT_THRESHOLD: i32 = 100;
/// 首尾行号之差不超过该值的文字带视为噪点，直接丢弃
const MIN_BAND_HEIGHT: usize = 5;
/// 文字行上方留出的空白行数
const BAND_PADDING: usize = 1;

/// 从模型元数据里推断出的输入/输出布局，避免把张量名写死
struct IoPlan {
    /// 图像输入在会话输入列表中的下标
    image_input: usize,
    /// 序列长度输入（一维张量）的下标；模型没有该输入时为 `None`
    length_input: Option<usize>,
    /// `[1, seq_len, num_classes]` 的 logits 输出下标
    logits_output: usize,
    /// 一维 int64 的 `output_lengths` 输出下标
    length_output: Option<usize>,
}

/// 按 C++ 侧 `onnxmodle` 的做法组装输入：一维输入填序列长度，其余填图像
fn build_plan(session: &Session) -> IoPlan {
    let mut image_input = None;
    let mut length_input = None;
    let mut logits_output = None;
    let mut length_output = None;

    for (index, input) in session.inputs().iter().enumerate() {
        match input.dtype() {
            ValueType::Tensor { shape, .. } if shape.len() == 1 => {
                if length_input.is_none() {
                    length_input = Some(index);
                }
            }
            ValueType::Tensor { .. } if image_input.is_none() => image_input = Some(index),
            ValueType::Tensor { .. } => panic!("识别模型包含多个图像输入，暂不支持"),
            _ => panic!("识别模型输入 #{index} 不是张量，无法处理")
        }
    }

    for (index, output) in session.outputs().iter().enumerate() {
        if let ValueType::Tensor { ty, shape, .. } = output.dtype() {
            match ty {
                TensorElementType::Float32 if shape.len() == 3 && logits_output.is_none() => {
                    logits_output = Some(index)
                }
                TensorElementType::Int64 if shape.len() == 1 => length_output = Some(index),
                _ => {}
            }
        }
    }

    IoPlan {
        image_input: image_input.expect("识别模型缺少图像输入"),
        length_input,
        logits_output: logits_output.expect("识别模型未返回 3 维 logits"),
        length_output
    }
}

/// 按行读取字典，一个 token 占一行；`<space>` 读成空格，`<unused*>` 占位标签读成空字符串
fn load_dict(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("读取字典文件 {} 失败: {}", path.display(), e));
    let text = String::from_utf8_lossy(&bytes);
    let mut tokens = Vec::new();
    for line in text.lines() {
        // Windows 下 checkout 出来的字典是 CRLF，`lines()` 已经去掉 '\r'
        let trimmed = line.trim_matches(|c| c == ' ' || c == '\t');
        tokens.push(if trimmed == "<space>" {
            " ".to_string()
        } else if trimmed.starts_with("<unused") {
            // cnocr 字典里的占位标签，输出为空字符串
            String::new()
        } else {
            line.to_string()
        });
    }
    tokens
}

pub struct Ocr {
    dict: Vec<String>,
    session: Mutex<Session>,
    plan: IoPlan,
    spec: &'static ModelSpec,
    blank_index: usize
}

impl Ocr {
    /// 使用默认模型（cnocr136fc）
    pub fn new() -> Ocr {
        Ocr::with_model(UseModel::default())
    }

    /// 使用注册表中的指定模型
    pub fn with_model(model: UseModel) -> Ocr {
        Ocr::with_spec(model.spec())
    }

    /// 按模型名选择，例如 `"ch_PP-OCRv3"`；未知名称会 panic
    pub fn with_model_name(name: &str) -> Ocr {
        let spec = find_recognizer(name)
            .unwrap_or_else(|| panic!("未注册的识别模型: {}（可用名称见 model_registry::RECOGNIZERS）", name));
        Ocr::with_spec(spec)
    }

    /// 全部可选模型
    pub fn available_models() -> &'static [ModelSpec] {
        crate::model_registry::RECOGNIZERS
    }

    /// 当前使用的模型
    pub fn model(&self) -> &'static ModelSpec {
        self.spec
    }

    fn with_spec(spec: &'static ModelSpec) -> Ocr {
        let model_file = model_path(spec);
        let session = Session::builder()
            .expect("创建 onnxruntime 会话失败")
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .expect("设置图优化级别失败")
            .commit_from_file(&model_file)
            .unwrap_or_else(|e| panic!("载入模型 {} 失败: {}", model_file.display(), e));
        let plan = build_plan(&session);

        let dict = load_dict(&dict_path(spec));
        if dict.is_empty() {
            panic!("字典文件为空: {}", spec.dict_file);
        }
        // cnocr 自训练模型把 blank 放在最后一类；PaddleOCR 系模型 blank 为第 0 类
        let blank_index = if spec.blank_index < 0 { dict.len() } else { spec.blank_index as usize };

        Ocr { dict, session: Mutex::new(session), plan, spec, blank_index }
    }

    /// 把单行灰度图缩放并归一化成模型输入（CHW 排列）
    fn prepare_line_image(&self, line: &GrayImage) -> (Vec<f32>, usize) {
        let height = self.spec.height;
        let width = (line.width() as usize * height / line.height() as usize).max(MIN_LINE_WIDTH);
        let resized = image::imageops::resize(line, width as u32, height as u32, FilterType::Triangle);

        let plane = width * height;
        let mut chw = vec![0f32; plane * self.spec.channels];
        if self.spec.channels == 1 {
            for (index, &pixel) in resized.as_raw().iter().enumerate() {
                chw[index] = pixel as f32 / 255.0;
            }
        } else {
            // 灰度复制成 3 通道后按 PaddleOCR 的方式归一化
            for c in 0..self.spec.channels {
                let offset = c * plane;
                for (i, &pixel) in resized.as_raw().iter().enumerate() {
                    let value = pixel as f32 / 255.0;
                    chw[offset + i] = (value - 0.5) / 0.5;
                }
            }
        }
        (chw, width)
    }

    fn recognize_line(&self, line: &GrayImage) -> (String, f32) {
        if line.width() < 2 || line.height() < 2 {
            return (String::new(), 0.0);
        }
        let (chw, width) = self.prepare_line_image(line);

        let mut session = self.session.lock().expect("识别会话状态异常");
        let input_count = session.inputs().len();
        let length = Tensor::from_array(([1usize], vec![width as i64])).expect("构造序列长度张量失败");
        let image = TensorRef::from_array_view(([1usize, self.spec.channels, self.spec.height, width], chw.as_slice()))
            .expect("构造输入张量失败");

        let mut inputs: Vec<SessionInputValue> = Vec::with_capacity(input_count);
        for index in 0..input_count {
            if index == self.plan.image_input {
                inputs.push((&*image).into());
            } else if Some(index) == self.plan.length_input {
                inputs.push((&length).into());
            } else {
                panic!("识别模型包含无法处理的输入 #{index}");
            }
        }

        let outputs = session.run(&inputs[..]).expect("模型推理失败");
        let (shape, logits) = outputs[self.plan.logits_output]
            .try_extract_tensor::<f32>()
            .expect("读取 logits 失败");
        let classes = *shape.last().expect("模型输出的 logits 形状异常") as usize;
        if classes == 0 || logits.len() < classes {
            return (String::new(), 0.0);
        }

        // 有效长度：模型给出的 output_lengths 优先，否则用整个序列
        let seq_len = logits.len() / classes;
        let valid_len = match self.plan.length_output {
            Some(index) => outputs[index]
                .try_extract_tensor::<i64>()
                .ok()
                .and_then(|(_, lengths)| lengths.first().copied())
                .map_or(seq_len, |len| (len.max(0) as usize).min(seq_len)),
            None => seq_len
        };

        self.decode_ctc(logits, classes, valid_len)
    }

    /// logits 按时间步展平（每步 `classes` 个类别），逐步 softmax 取 argmax 后做 CTC 去重
    fn decode_ctc(&self, logits: &[f32], classes: usize, valid_len: usize) -> (String, f32) {
        let steps = logits.len().div_euclid(classes).min(valid_len);
        let mut argmax = Vec::with_capacity(steps);
        let mut confidence = 1.0f32;
        for step in 0..steps {
            let row = &logits[step * classes..step * classes + classes];
            let (best, max_value) = row.iter().enumerate().fold((0usize, row[0]), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
            // 归一化后该步的最大概率即 1/Σexp(x-max)
            let sum: f64 = row.iter().map(|x| (f64::from(*x) - f64::from(max_value)).exp()).sum();
            confidence = confidence.min((1.0 / sum) as f32);
            argmax.push(best);
        }
        (self.ctc_best(&argmax), confidence)
    }
}

impl Default for Ocr {
    fn default() -> Ocr {
        Ocr::new()
    }
}

pub trait OcrTraitConst {
    fn from_image(&self, img: &DynamicImage) -> Vec<(String, f32)>;
    fn from_path(&self, path: &str) -> Vec<(String, f32)>;
    fn ocr_for_single_lines(&self, inimgs: &[GrayImage]) -> Vec<(String, f32)>;
    fn line_split(&self, inimg: &GrayImage) -> Vec<GrayImage>;
    fn ctc_best(&self, data: &[usize]) -> String;
}

impl OcrTraitConst for Ocr {
    /**
    # 从图片识别文字
    > 输入： 彩色或灰度图片
    输出 识别出来的文字 和 准确率
    */
    fn from_image(&self, img: &DynamicImage) -> Vec<(String, f32)> {
        let mut gray = img.to_luma8();
        let imgcol = gray.width() as usize;
        let imgrow = gray.height() as usize;
        if std::cmp::min(imgcol, imgrow) < 2 {
            return Vec::new();
        }
        // 按第一列像素和判断底色，深底白字先反色成白底黑字
        let col_sum: u64 = gray.as_raw().chunks(imgcol).map(|row| row[0] as u64).sum();
        if col_sum < 145 {
            for p in gray.as_mut().iter_mut() {
                *p = 255 - *p;
            }
        }
        let imgs = self.line_split(&gray);
        self.ocr_for_single_lines(&imgs)
    }
    /**
    # 从文件路径 读取图片 识别文字
    > 输入： 因该是 图片文件路径
    输出 识别出来的文字 和 准确率
    */
    fn from_path(&self, path: &str) -> Vec<(String, f32)> {
        let img = image::open(path).unwrap_or_else(|e| panic!("读取图片 {} 失败: {}", path, e));
        self.from_image(&img)
    }
    /**
    # 识别单行文字
    > 输入： 因该是一组 切分好行 的灰度图片
    输出 识别出来的文字 和 准确率
    */
    fn ocr_for_single_lines(&self, inimgs: &[GrayImage]) -> Vec<(String, f32)> {
        inimgs.iter().map(|line| self.recognize_line(line)).collect()
    }
    /**
    # line_split
    > 按行取反色后的最大像素值，超过阈值的连续行判为文字行并切出子图。
    > 所以 输入图片不能有 划线 类似表格等 如有倾斜也会导致无法划分图片切片
    输出 切分好的 图片 列表
    */
    fn line_split(&self, inimg: &GrayImage) -> Vec<GrayImage> {
        let mut list: Vec<GrayImage> = Vec::new();
        let imgcol = inimg.width() as usize;
        let imgrow = inimg.height() as usize;
        if imgcol == 0 || imgrow == 0 {
            return list;
        }

        let raw = inimg.as_raw();
        let mut band: Option<(usize, usize)> = None;
        for y in 0..imgrow {
            let row = &raw[y * imgcol..(y + 1) * imgcol];
            // 反色后的行最大值 = 255 - 原图行最小值
            let inverted_max = 255 - *row.iter().min().unwrap_or(&255) as i32;
            if inverted_max > ROW_TEXT_THRESHOLD {
                band = Some(match band {
                    Some((start, _)) => (start, y),
                    None => (y, y)
                });
            } else if let Some((start, end)) = band.take() {
                push_band(&mut list, inimg, start, end);
            }
        }
        if let Some((start, end)) = band.take() {
            push_band(&mut list, inimg, start, end);
        }
        list
    }
    /**
    #合成 文字
    > 输入：每步 argmax 的类别下标列表
    > 功能：跳过 blank、去掉与上一步相同的下标，再用字典把剩余下标换成文字
    输出 处理好的 字符串
    */
    fn ctc_best(&self, data: &[usize]) -> String {
        let mut res = String::new();
        let mut previous: Option<usize> = None;
        for &index in data {
            if Some(index) != previous && index != self.blank_index && index < self.dict.len() {
                res.push_str(&self.dict[index]);
            }
            previous = Some(index);
        }
        res
    }
}

/// 文字带高度不足时丢弃，否则行首补一行空白后切出子图；`end` 为闭区间末行
fn push_band(list: &mut Vec<GrayImage>, img: &GrayImage, start: usize, end: usize) {
    if end - start <= MIN_BAND_HEIGHT {
        return;
    }
    let top = start.saturating_sub(BAND_PADDING);
    let bottom = (end + 1).min(img.height() as usize);
    if bottom > top {
        let crop = image::imageops::crop_imm(img, 0, top as u32, img.width(), (bottom - top) as u32);
        list.push(crop.to_image());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 这些用例需要仓库根目录下的 `cnocr136fc.onnx` 与 `label_cn.txt`
    fn ocr() -> Ocr {
        Ocr::new()
    }

    fn striped(rows: &[std::ops::Range<usize>], width: usize, height: usize) -> GrayImage {
        let mut data = vec![255u8; width * height];
        for range in rows {
            for y in range.clone() {
                data[y * width..(y + 1) * width].fill(0);
            }
        }
        GrayImage::from_raw(width as u32, height as u32, data).expect("构造测试图片失败")
    }

    #[test]
    fn recognizes_sample_image() {
        let lines = ocr().from_path("cnocr.png");
        assert_eq!(lines.len(), 2, "示例图片应有两行文字");
        assert!(lines[0].0.starts_with("cnocr自V2.1.2之后"), "第一行识别结果异常: {}", lines[0].0);
        assert!(lines[1].0.starts_with("ONNX版本"), "第二行识别结果异常: {}", lines[1].0);
        for (_, confidence) in &lines {
            assert!(*confidence > 0.4 && *confidence <= 1.0, "置信度超出范围: {}", confidence);
        }
    }

    #[test]
    fn line_split_keeps_only_tall_enough_bands() {
        // 0..10 行是文字带，20..22 行太矮会被丢弃
        let img = striped(&[0..10, 20..22], 40, 40);
        let bands = ocr().line_split(&img);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].height(), 10);
        assert_eq!(bands[0].width(), 40);
    }

    #[test]
    fn handles_small_and_blank_images() {
        let ocr = ocr();
        assert!(ocr.from_image(&DynamicImage::new_rgb8(1, 1)).is_empty());
        // 纯黑图片会先反色成纯白，切不出文字行
        assert!(ocr.from_image(&DynamicImage::new_luma8(400, 60)).is_empty());
        // 纯白图片本身就没有文字行
        let white = GrayImage::from_pixel(400, 60, image::Luma([255]));
        assert!(ocr.line_split(&white).is_empty());
    }
}
