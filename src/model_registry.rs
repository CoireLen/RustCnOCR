//! 识别模型清单与本地文件定位，参考 CnocrCpp 的 `model_registry.h/.cpp`。
//!
//! 模型文件依次查找 `modle/`、当前目录、缓存目录（`CNOCR_HOME`，未设置时 Windows 用
//! `%APPDATA%\cnocr`、其他平台用 `~/.cnocr`，再拼上 `models`）；字典文件只查找 `dict/` 和当前目录。
//! 本 crate 只做识别，不含 CnocrCpp 的 PP-OCRv3 文本行检测（`cnstd`），也不会联网下载模型。

use std::path::PathBuf;

/// 单个识别模型的推理参数与配套文件
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// 模型名，与 Python cnocr / CnocrCpp 保持一致
    pub name: &'static str,
    /// 输入通道：1 灰度（`x/255`）；3 彩色（`(x/255-0.5)/0.5`）
    pub channels: usize,
    /// 输入高度：cnocr 自训练模型与 PaddleOCR mobile 系为 32，PP-OCRv3 为 48
    pub height: usize,
    /// CTC blank 下标；负数表示 blank 是最后一类（即字典长度）
    pub blank_index: i32,
    /// 模型文件名
    pub onnx_file: &'static str,
    /// 字典文件名
    pub dict_file: &'static str,
    /// 模型用途说明
    pub description: &'static str,
}

/// 全部可用的识别模型
pub const RECOGNIZERS: &[ModelSpec] = &[
    ModelSpec {
        name: "cnocr136fc",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "cnocr136fc.onnx",
        dict_file: "label_cn.txt",
        description: "早期中文识别模型（cnocr 2.2）"
    },
    ModelSpec {
        name: "densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "通用中文识别模型（cnocr 2.3）"
    },
    ModelSpec {
        name: "scene-densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "scene-densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "场景文字识别模型（cnocr 2.3）"
    },
    ModelSpec {
        name: "doc-densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "doc-densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "文档识别模型（cnocr 2.3）"
    },
    ModelSpec {
        name: "number-densenet_lite_136-fc",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "number-densenet_lite_136-fc.onnx",
        dict_file: "label_number.txt",
        description: "数字识别模型（cnocr 2.3）"
    },
    ModelSpec {
        name: "en_number_mobile_v2.0",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "en_number_mobile_v2.0_rec_infer.onnx",
        dict_file: "en_dict.txt",
        description: "英文数字识别模型（PaddleOCR）"
    },
    ModelSpec {
        name: "chinese_cht_PP-OCRv3",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "chinese_cht_PP-OCRv3_rec_infer.onnx",
        dict_file: "chinese_cht_dict.txt",
        description: "繁体中文识别模型（PaddleOCR PP-OCRv3）"
    },
    ModelSpec {
        name: "ch_PP-OCRv3",
        channels: 3,
        height: 48,
        blank_index: 0,
        onnx_file: "ch_PP-OCRv3_rec_infer.onnx",
        dict_file: "ppocr_keys_v1.txt",
        description: "简体中英文识别模型（PaddleOCR PP-OCRv3）"
    },
    ModelSpec {
        name: "ch_ppocr_mobile_v2.0",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "ch_ppocr_mobile_v2.0_rec_infer.onnx",
        dict_file: "ppocr_keys_v1.txt",
        description: "简体中英文识别模型（PaddleOCR mobile v2.0）"
    },
];

/// 按模型名查找注册项
pub fn find_recognizer(name: &str) -> Option<&'static ModelSpec> {
    RECOGNIZERS.iter().find(|spec| spec.name == name)
}

/// 内置识别模型选择，对应 CnocrCpp 的 `cnocr::USE_MODLE`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UseModel {
    #[default]
    Cnocr136fc,
    DensenetLite136Gru,
    SceneDensenet136Gru,
    DocDensenet136Gru,
    NumberDensenet136Fc,
    EnNumber,
    ChineseCht,
    ChPpOcrV3,
    ChPpOcrMobileV2,
}

impl UseModel {
    /// 模型名（与 [`RECOGNIZERS`] 中的 `name` 一致）
    pub fn name(self) -> &'static str {
        match self {
            UseModel::Cnocr136fc => "cnocr136fc",
            UseModel::DensenetLite136Gru => "densenet_lite_136-gru",
            UseModel::SceneDensenet136Gru => "scene-densenet_lite_136-gru",
            UseModel::DocDensenet136Gru => "doc-densenet_lite_136-gru",
            UseModel::NumberDensenet136Fc => "number-densenet_lite_136-fc",
            UseModel::EnNumber => "en_number_mobile_v2.0",
            UseModel::ChineseCht => "chinese_cht_PP-OCRv3",
            UseModel::ChPpOcrV3 => "ch_PP-OCRv3",
            UseModel::ChPpOcrMobileV2 => "ch_ppocr_mobile_v2.0",
        }
    }

    pub fn spec(self) -> &'static ModelSpec {
        find_recognizer(self.name())
            .unwrap_or_else(|| panic!("模型注册表缺少 {}，请检查 model_registry.rs", self.name()))
    }
}

fn env_dir(key: &str) -> Option<PathBuf> {
    let value = std::env::var_os(key)?;
    let path = PathBuf::from(value);
    (!path.as_os_str().is_empty()).then_some(path)
}

/// 模型缓存根目录，对应 CnocrCpp 的 `modelfiles::data_dir()`
fn data_dir() -> PathBuf {
    if let Some(dir) = env_dir("CNOCR_HOME") {
        return dir;
    }
    #[cfg(windows)]
    {
        if let Some(dir) = env_dir("APPDATA") {
            return dir.join("cnocr");
        }
        if let Some(dir) = env_dir("USERPROFILE") {
            return dir.join(".cnocr");
        }
    }
    #[cfg(not(windows))]
    if let Some(dir) = env_dir("HOME") {
        return dir.join(".cnocr");
    }
    std::env::temp_dir().join("cnocr")
}

/// 模型缓存目录（`<data_dir>/models`）
fn cache_dir() -> PathBuf {
    data_dir().join("models")
}

/// 依次尝试各目录下的 `<dir>/<file>`（空目录即当前目录），返回第一个存在的文件
fn locate(file: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    dirs.iter().map(|dir| dir.join(file)).find(|path| path.is_file())
}

/// 模型文件路径；缺失时 panic 并列出查找过的位置
pub fn model_path(spec: &ModelSpec) -> PathBuf {
    let cache = cache_dir();
    locate(spec.onnx_file, &[PathBuf::from("modle"), PathBuf::new(), cache.clone()]).unwrap_or_else(|| {
        panic!(
            "未找到模型文件 {}，请放到 modle/、当前目录或 {}（模型可从 https://hf-mirror.com/breezedeus/cnstd-cnocr-models 下载）",
            spec.onnx_file,
            cache.display()
        )
    })
}

/// 字典文件路径；缺失时 panic 并列出查找过的位置
pub fn dict_path(spec: &ModelSpec) -> PathBuf {
    locate(spec.dict_file, &[PathBuf::from("dict"), PathBuf::new()]).unwrap_or_else(|| {
        panic!("未找到字典文件 {}，请放到 dict/ 或当前目录", spec.dict_file)
    })
}
