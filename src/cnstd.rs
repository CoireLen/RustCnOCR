//! DBNet 文本行检测（对应 CnocrCpp 的 `cnstd` + `db_postprocess`），
//! 使用 PP-OCRv3 检测模型，输入灰度图，输出按阅读顺序排列的文本框。

use std::sync::Mutex;

use image::imageops::FilterType;
use image::GrayImage;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{TensorRef, ValueType};

use crate::geometry::{
    mini_boxes, offset_polygon, order_points_clockwise, point_distance, polygon_area,
    polygon_perimeter, sort_reading_order, DetBox, Pt
};
use crate::model_registry::{det_model_spec, ensure_model};

/// 概率图尺寸的对齐倍数，与 Python cnstd 的 `SizeDivisor` 一致
const SIZE_DIVISOR: u32 = 32;
/// 一次处理的最大轮廓数
const MAX_CANDIDATES: usize = 1000;

/// DBNet 后处理参数，取值与 `db_postprocess.h` 的默认值一致
#[derive(Clone, Copy, Debug)]
struct DbParams {
    /// 概率图二值化阈值
    bin_thresh: f32,
    /// 文本框得分阈值
    box_thresh: f32,
    /// 文本框外扩比例
    unclip_ratio: f32,
    /// 最小文本框边长
    min_box_size: usize,
}

/// 检测参数
#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// 缩放后送入模型的最长边
    pub long_side: u32,
    /// 文本框得分阈值
    pub box_score_thresh: f32,
    /// 文本框外扩比例，行尾标点被裁掉时调大该值
    pub unclip_ratio: f32,
    /// 最小文本框边长
    pub min_box_size: usize,
}

impl Default for DetectOptions {
    fn default() -> DetectOptions {
        DetectOptions { long_side: 768, box_score_thresh: 0.3, unclip_ratio: 1.5, min_box_size: 4 }
    }
}

/// 等比缩放到长边不超过 `long_side`，并把宽高对齐到 32 的倍数
fn resized_shape(src_width: u32, src_height: u32, long_side: u32) -> (u32, u32) {
    let ratio = f64::from(long_side) / f64::from(src_width.max(src_height).max(1));
    let new_width = (f64::from(src_width) * ratio).max(1.0) as u32;
    let new_height = (f64::from(src_height) * ratio).max(1.0) as u32;
    let calibrate = |value: u32| (f64::from(value) / f64::from(SIZE_DIVISOR)).round() as u32 * SIZE_DIVISOR;
    (calibrate(new_width).max(SIZE_DIVISOR), calibrate(new_height).max(SIZE_DIVISOR))
}

/// 灰度图缩放到目标尺寸后做 ImageNet 归一化，输出 CHW 排列的 float 数据
fn fill_normalized_chw(img: &GrayImage, width: u32, height: u32) -> Vec<f32> {
    let resized = image::imageops::resize(img, width, height, FilterType::Triangle);
    let mean = [0.485f32, 0.456, 0.406];
    let stddev = [0.229f32, 0.224, 0.225];
    let plane = (width * height) as usize;
    let mut chw = vec![0f32; plane * 3];
    for c in 0..3 {
        let offset = c * plane;
        for (i, &pixel) in resized.as_raw().iter().enumerate() {
            let value = pixel as f32 / 255.0;
            chw[offset + i] = (value - mean[c]) / stddev[c];
        }
    }
    chw
}

/// 8 连通前景区域标注，返回每个区域的边界像素（等价于 `cv::findContours` 的外轮廓点集）
///
/// 与 opencv 的差别：不单独返回内孔轮廓。DBNet 后处理只用到点集的最小外接矩形，
/// 内孔不会改变外接矩形，因此对结果没有影响。
fn connected_components(binary: &[u8], width: usize, height: usize) -> Vec<Vec<Pt>> {
    let index = |x: usize, y: usize| y * width + x;
    let is_text = |x: usize, y: usize| x < width && y < height && binary[index(x, y)] != 0;

    let mut visited = vec![false; width * height];
    let mut regions: Vec<Vec<Pt>> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            if !is_text(start_x, start_y) || visited[index(start_x, start_y)] {
                continue;
            }
            let mut pixels: Vec<(usize, usize)> = Vec::new();
            stack.push((start_x, start_y));
            visited[index(start_x, start_y)] = true;
            while let Some((x, y)) = stack.pop() {
                pixels.push((x, y));
                for (dx, dy) in [(-1i32, -1i32), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (x as i64 + i64::from(dx), y as i64 + i64::from(dy));
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if is_text(nx, ny) && !visited[index(nx, ny)] {
                        visited[index(nx, ny)] = true;
                        stack.push((nx, ny));
                    }
                }
            }

            // 只保留边界像素：四邻域里存在非前景
            let boundary = pixels
                .into_iter()
                .filter(|(x, y)| {
                    !is_text(x.wrapping_sub(1), *y)
                        || !is_text(x + 1, *y)
                        || (*y == 0 || !is_text(*x, y - 1))
                        || !is_text(*x, y + 1)
                })
                .map(|(x, y)| Pt::new(x as f32, y as f32))
                .collect::<Vec<_>>();
            if !boundary.is_empty() {
                regions.push(boundary);
            }
        }
    }
    regions
}

/// 框内平均概率，对应 `box_score_fast`
fn box_score_fast(prob: &[f32], prob_width: usize, prob_height: usize, box_points: &[Pt; 4]) -> f32 {
    if prob_width == 0 || prob_height == 0 {
        return 0.0;
    }
    let min_x = box_points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max_x = box_points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let min_y = box_points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = box_points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

    let xmin = (min_x.floor().max(0.0) as usize).min(prob_width - 1);
    let xmax = (max_x.ceil().max(0.0) as usize).min(prob_width - 1);
    let ymin = (min_y.floor().max(0.0) as usize).min(prob_height - 1);
    let ymax = (max_y.ceil().max(0.0) as usize).min(prob_height - 1);
    if xmax < xmin || ymax < ymin {
        return 0.0;
    }

    let roi_width = xmax - xmin + 1;
    let roi_height = ymax - ymin + 1;
    let local: Vec<Pt> = box_points
        .iter()
        .map(|p| Pt::new((p.x - xmin as f32).floor().max(0.0), (p.y - ymin as f32).floor().max(0.0)))
        .collect();

    let mut mask = vec![0u8; roi_width * roi_height];
    crate::geometry::fill_polygon(&[local[0], local[1], local[2], local[3]], &mut mask, roi_width, roi_height);

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..roi_height {
        for x in 0..roi_width {
            if mask[y * roi_width + x] == 1 {
                sum += f64::from(prob[(ymin + y) * prob_width + xmin + x]);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// 概率图转文本框（坐标已还原到原图，并按阅读顺序排序）
fn boxes_from_prob(
    prob: &[f32],
    prob_width: usize,
    prob_height: usize,
    src_width: usize,
    src_height: usize,
    params: &DbParams
) -> Vec<DetBox> {
    let mut boxes: Vec<DetBox> = Vec::new();
    if prob.len() < prob_width * prob_height || prob_width == 0 || prob_height == 0 || src_width < 2 || src_height < 2 {
        return boxes;
    }

    let binary: Vec<u8> = prob.iter().map(|&p| if p > params.bin_thresh { 255 } else { 0 }).collect();
    let regions = connected_components(&binary, prob_width, prob_height);

    for contour in regions.into_iter().take(MAX_CANDIDATES) {
        if contour.len() < 4 {
            continue;
        }
        let Some((box_points, side)) = mini_boxes(&contour) else { continue };
        if side < 3.0 {
            continue;
        }

        let score = box_score_fast(prob, prob_width, prob_height, &box_points);
        if score < params.box_thresh {
            continue;
        }

        let area = polygon_area(&box_points);
        let perimeter = polygon_perimeter(&box_points);
        let expand = (area * f64::from(params.unclip_ratio) / perimeter.max(1e-6)) as f32;
        let expanded = offset_polygon(&box_points, expand);
        let Some((expanded_box, expanded_side)) = mini_boxes(&expanded) else { continue };
        if expanded_side < 5.0 {
            continue;
        }

        // 概率图坐标还原到原图
        let mut scaled = [Pt::default(); 4];
        for (k, point) in expanded_box.iter().enumerate() {
            scaled[k] = Pt::new(
                (point.x / prob_width as f32 * src_width as f32).round().clamp(0.0, src_width as f32),
                (point.y / prob_height as f32 * src_height as f32).round().clamp(0.0, src_height as f32)
            );
        }

        let mut ordered = order_points_clockwise(&scaled);
        for point in &mut ordered {
            point.x = point.x.clamp(0.0, (src_width - 1) as f32);
            point.y = point.y.clamp(0.0, (src_height - 1) as f32);
        }

        let box_width = point_distance(ordered[0], ordered[1]) as usize;
        let box_height = point_distance(ordered[0], ordered[3]) as usize;
        if box_width.min(box_height) < params.min_box_size {
            continue;
        }
        boxes.push(DetBox { points: ordered, score });
    }

    sort_reading_order(&mut boxes);
    boxes
}

/// PP-OCRv3 文本行检测器
pub struct CnStd {
    session: Mutex<Session>,
    input_count: usize,
}

impl CnStd {
    /// 载入检测模型；本地没有时从镜像自动下载
    pub fn new() -> CnStd {
        CnStd::with_auto_download(true)
    }

    /// 载入检测模型；`auto_download` 为假且本地找不到模型时 panic
    pub fn with_auto_download(auto_download: bool) -> CnStd {
        let path = ensure_model(det_model_spec(), auto_download);
        let mut builder = Session::builder()
            .expect("创建 onnxruntime 会话失败")
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .expect("设置图优化级别失败");
        let session = builder
            .commit_from_file(&path)
            .unwrap_or_else(|e| panic!("载入检测模型 {} 失败: {}", path.display(), e));
        let input_count = session.inputs().len();
        if input_count != 1 {
            panic!("检测模型应有 1 个输入，实际 {} 个", input_count);
        }
        CnStd { session: Mutex::new(session), input_count }
    }

    /// 检测图片中的文本行
    pub fn detect(&self, img: &GrayImage, options: &DetectOptions) -> Vec<DetBox> {
        let src_width = img.width() as usize;
        let src_height = img.height() as usize;
        if src_width < 2 || src_height < 2 {
            return Vec::new();
        }

        let (target_width, target_height) = resized_shape(img.width(), img.height(), options.long_side);
        let chw = fill_normalized_chw(img, target_width, target_height);

        let mut session = self.session.lock().expect("检测会话状态异常");
        let input = TensorRef::from_array_view(
            ([1usize, 3, target_height as usize, target_width as usize], chw.as_slice())
        )
        .expect("构造检测输入张量失败");
        let mut inputs: Vec<SessionInputValue> = Vec::with_capacity(self.input_count);
        inputs.push((&*input).into());

        let outputs = session.run(&inputs[..]).expect("检测模型推理失败");
        let output_index = find_prob_output(&outputs);
        let (shape, prob) = outputs[output_index]
            .try_extract_tensor::<f32>()
            .expect("读取检测概率图失败");
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.len() != 4 {
            panic!("检测模型概率图形状异常: {:?}", dims);
        }
        let (prob_height, prob_width) = (dims[2], dims[3]);

        let params = DbParams {
            bin_thresh: 0.3,
            box_thresh: options.box_score_thresh,
            unclip_ratio: options.unclip_ratio,
            min_box_size: options.min_box_size
        };
        boxes_from_prob(prob, prob_width, prob_height, src_width, src_height, &params)
    }
}

impl Default for CnStd {
    fn default() -> CnStd {
        CnStd::new()
    }
}

/// 找到 `[1, 1, height, width]` 的概率图输出
fn find_prob_output(outputs: &ort::session::SessionOutputs) -> usize {
    for index in 0..outputs.len() {
        if let ValueType::Tensor { shape, .. } = outputs[index].dtype() {
            if shape.len() == 4 {
                return index;
            }
        }
    }
    panic!("检测模型未返回 4 维概率图");
}

/// 按四点透视变换裁剪文本行，必要时逆时针旋转成水平
pub fn crop_line(img: &GrayImage, box_points: &DetBox) -> Option<GrayImage> {
    let points = &box_points.points;
    let crop_width = point_distance(points[0], points[1])
        .max(point_distance(points[3], points[2]))
        .floor() as u32;
    let crop_height = point_distance(points[0], points[3])
        .max(point_distance(points[1], points[2]))
        .floor() as u32;
    if crop_width == 0 || crop_height == 0 {
        return None;
    }

    let destination = [
        Pt::new(0.0, 0.0),
        Pt::new(crop_width as f32, 0.0),
        Pt::new(crop_width as f32, crop_height as f32),
        Pt::new(0.0, crop_height as f32)
    ];
    let transform = perspective_transform(points, &destination)?;
    let mut cropped = GrayImage::from_pixel(crop_width, crop_height, image::Luma([255]));
    for y in 0..crop_height {
        for x in 0..crop_width {
            let (sx, sy) = transform.map_inverse(f64::from(x), f64::from(y));
            cropped.put_pixel(x, y, image::Luma([sample_replicate(img, sx, sy)]));
        }
    }

    // 竖排文本旋转成横排
    if f64::from(cropped.height()) / f64::from(cropped.width()) >= 1.5 {
        return Some(image::imageops::rotate270(&cropped));
    }
    Some(cropped)
}

/// 双线性采样，越界按 `BORDER_REPLICATE` 处理
fn sample_replicate(img: &GrayImage, x: f64, y: f64) -> u8 {
    let (width, height) = (img.width() as f64, img.height() as f64);
    let clamp = |value: f64, max: f64| value.max(0.0).min(max - 1.0);
    let (x, y) = (clamp(x, width), clamp(y, height));
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(img.width() - 1);
    let y1 = (y0 + 1).min(img.height() - 1);
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let get = |x: u32, y: u32| f64::from(img.get_pixel(x, y)[0]);
    let top = get(x0, y0) * (1.0 - fx) + get(x1, y0) * fx;
    let bottom = get(x0, y1) * (1.0 - fx) + get(x1, y1) * fx;
    (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8
}

/// 单应矩阵：把 `source` 四点映射到 `destination` 四点，与 `cv::getPerspectiveTransform` 等价
#[derive(Clone, Copy, Debug)]
struct Homography {
    /// 逆变换用的行优先 3x3 矩阵，采样时把目标像素坐标映回源图
    inverse: [f64; 9]
}

impl Homography {
    /// 目标像素坐标反算源图坐标
    fn map_inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let i = &self.inverse;
        let w = i[6] * x + i[7] * y + i[8];
        if w.abs() < 1e-12 {
            return (x, y);
        }
        ((i[0] * x + i[1] * y + i[2]) / w, (i[3] * x + i[4] * y + i[5]) / w)
    }
}

fn perspective_transform(source: &[Pt; 4], destination: &[Pt; 4]) -> Option<Homography> {
    // 每个点对给出两条例性方程，未知数为 h11..h32（h33 = 1）
    let mut rows = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (f64::from(source[i].x), f64::from(source[i].y));
        let (u, v) = (f64::from(destination[i].x), f64::from(destination[i].y));
        rows[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        rows[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }

    // 带部分主元的高斯消元，主元按绝对值选取
    for col in 0..8 {
        let pivot =
            (col..8).max_by(|a, b| rows[*a][col].abs().total_cmp(&rows[*b][col].abs())).unwrap();
        if rows[pivot][col].abs() < 1e-12 {
            return None;
        }
        rows.swap(col, pivot);
        let divisor = rows[col][col];
        for j in col..9 {
            rows[col][j] /= divisor;
        }
        for r in 0..8 {
            if r != col && rows[r][col].abs() > 0.0 {
                let factor = rows[r][col];
                for j in col..9 {
                    rows[r][j] -= factor * rows[col][j];
                }
            }
        }
    }

    let mut m = [0.0f64; 8];
    for (i, item) in m.iter_mut().enumerate() {
        *item = rows[i][8];
    }
    let full = [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], 1.0];
    invert3x3(&full).map(|inverse| Homography { inverse })
}

/// 3x3 矩阵求逆，奇异矩阵返回 `None`
fn invert3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let cofactor = |i: usize, j: usize| -> f64 {
        let minor: Vec<f64> = (0..3)
            .flat_map(|r| (0..3).filter(move |c| r != i && *c != j).map(move |c| m[r * 3 + c]))
            .collect();
        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        sign * (minor[0] * minor[3] - minor[1] * minor[2])
    };
    let det = m[0] * cofactor(0, 0) + m[1] * cofactor(0, 1) + m[2] * cofactor(0, 2);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            // 伴随矩阵转置后除以行列式
            out[j * 3 + i] = cofactor(i, j) / det;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_shape_aligns_to_multiples_of_32() {
        assert_eq!(resized_shape(800, 600, 768), (768, 576));
        assert_eq!(resized_shape(100, 100, 768), (768, 768), "小图会等比放大到长边 768");
        assert_eq!(resized_shape(30, 1000, 768), (32, 768));
    }

    /// 在概率图上画一个矩形，检查后处理能还原出接近的文本框
    fn rectangle_prob(x0: usize, y0: usize, x1: usize, y1: usize) -> (Vec<f32>, usize, usize) {
        let (width, height) = (160usize, 80usize);
        let mut prob = vec![0f32; width * height];
        for y in y0..y1 {
            for x in x0..x1 {
                prob[y * width + x] = 0.9;
            }
        }
        (prob, width, height)
    }

    #[test]
    fn boxes_from_prob_finds_rectangles() {
        let params = DbParams { bin_thresh: 0.3, box_thresh: 0.3, unclip_ratio: 1.5, min_box_size: 4 };
        // 概率图上 20..120 x 20..40 的文本块，原图是两倍大
        let (prob, width, height) = rectangle_prob(20, 20, 120, 40);
        let boxes = boxes_from_prob(&prob, width, height, width * 2, height * 2, &params);
        assert_eq!(boxes.len(), 1, "应检测出一个文本框，实际 {}", boxes.len());
        let points = boxes[0].points;
        // 概率图坐标放大两倍回到原图，外扩让框比原始矩形（40..240, 40..80）各方向略大
        assert!(points[0].x > 8.0 && points[0].x < 40.0, "左上 x 异常: {:?}", points);
        assert!(points[0].y > 8.0 && points[0].y < 40.0, "左上 y 异常: {:?}", points);
        assert!(points[1].x > 240.0 && points[1].x < 280.0, "右上 x 异常: {:?}", points);
        assert!(points[2].y > 80.0 && points[2].y < 120.0, "右下 y 异常: {:?}", points);
        assert!(boxes[0].score > 0.5, "得分异常: {}", boxes[0].score);
    }

    #[test]
    fn boxes_from_prob_keeps_reading_order() {
        let params = DbParams { bin_thresh: 0.3, box_thresh: 0.3, unclip_ratio: 1.5, min_box_size: 4 };
        let (width, height) = (160usize, 80usize);
        let mut prob = vec![0f32; width * height];
        // 先画第二行，再画第一行，验证输出按阅读顺序
        for y in 50..65 {
            for x in 30..60 {
                prob[y * width + x] = 0.9;
            }
        }
        for y in 50..65 {
            for x in 90..120 {
                prob[y * width + x] = 0.9;
            }
        }
        let boxes = boxes_from_prob(&prob, width, height, width, height, &params);
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].points[0].x < boxes[1].points[0].x, "同一行应按左右排序");
    }

    #[test]
    fn crop_line_maps_box_to_horizontal_strip() {
        let mut img = GrayImage::from_pixel(60, 40, image::Luma([255]));
        for y in 10..20 {
            for x in 10..50 {
                img.put_pixel(x, y, image::Luma([0]));
            }
        }
        let box_points = DetBox {
            points: [Pt::new(10.0, 10.0), Pt::new(50.0, 10.0), Pt::new(50.0, 20.0), Pt::new(10.0, 20.0)],
            score: 1.0
        };
        let cropped = crop_line(&img, &box_points).expect("裁剪失败");
        assert_eq!(cropped.width(), 40);
        assert!(cropped.height() >= 10);
        assert!(cropped.iter().any(|&p| p < 128), "裁剪结果应包含文字像素");
    }

    /// 检测框常有 1~2 像素的上下边倾斜，单应矩阵不能因此退化为不可解
    #[test]
    fn crop_line_handles_slightly_sloped_box() {
        let mut img = GrayImage::from_pixel(400, 60, image::Luma([255]));
        for y in 20..35 {
            for x in 20..300 {
                img.put_pixel(x, y, image::Luma([0]));
            }
        }
        let box_points = DetBox {
            points: [
                Pt::new(20.0, 19.0),
                Pt::new(300.0, 21.0),
                Pt::new(300.0, 36.0),
                Pt::new(20.0, 34.0)
            ],
            score: 1.0
        };
        let cropped = crop_line(&img, &box_points).expect("倾斜文本框裁剪失败");
        assert!(cropped.width() > 270 && cropped.height() >= 15, "裁剪尺寸异常: {}x{}", cropped.width(), cropped.height());
        assert!(cropped.iter().any(|&p| p < 128), "裁剪结果应包含文字像素");
    }

    #[test]
    fn perspective_transform_round_trip() {
        let source = [Pt::new(0.0, 0.0), Pt::new(10.0, 0.0), Pt::new(10.0, 10.0), Pt::new(0.0, 10.0)];
        let destination = [Pt::new(1.0, 2.0), Pt::new(11.0, 2.0), Pt::new(11.0, 12.0), Pt::new(1.0, 12.0)];
        let transform = perspective_transform(&source, &destination).expect("求解单应矩阵失败");
        let (x, y) = transform.map_inverse(6.0, 7.0);
        assert!((x - 5.0).abs() < 1e-6 && (y - 5.0).abs() < 1e-6, "平移映射异常: {}, {}", x, y);
    }

    /// 用真实的 PP-OCRv3 检测模型跑一遍完整检测（需要 `ch_PP-OCRv3_det_infer.onnx`）
    #[test]
    fn detects_lines_in_sample_image() {
        let img = image::open("cnocr.png").expect("读取示例图片失败").to_luma8();
        let boxes = CnStd::new().detect(&img, &DetectOptions::default());
        assert_eq!(boxes.len(), 2, "示例图片应有两行文本，实际 {}", boxes.len());
        for window in boxes.windows(2) {
            assert!(window[0].points[0].y < window[1].points[0].y, "文本框未按从上到下排序: {:?}", boxes);
        }
        for line in &boxes {
            assert!(line.score > 0.3, "文本框得分过低: {}", line.score);
            let crop = crop_line(&img, line).expect("裁剪文本框失败");
            assert!(crop.width() > crop.height(), "文本行应是横条: {}x{}", crop.width(), crop.height());
        }
    }
}
