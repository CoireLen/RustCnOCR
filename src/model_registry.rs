//! 识别模型清单与本地文件定位，参考 CnocrCpp 的 `model_registry.h/.cpp`。
//!
//! 模型文件依次查找 `modle/`、当前目录、缓存目录（`CNOCR_HOME`，未设置时 Windows 用
//! `%APPDATA%\cnocr`、其他平台用 `~/.cnocr`，再拼上 `models`）；本地都没有时可以从
//! `breezedeus/cnstd-cnocr-models` 自动下载到缓存目录（见 [`crate::download`]）。
//! 字典文件只查找 `dict/` 和当前目录，镜像仓库不提供字典。

use std::path::PathBuf;

use crate::download;

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
    /// 自动下载时模型压缩包在仓库里的子目录，例如 `models/cnocr/2.3`
    pub subfolder: &'static str,
    /// 自动下载使用的压缩包名；为空表示没有可下载的包
    pub archive: &'static str,
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
        description: "早期中文识别模型（cnocr 2.2）",
        subfolder: "models/cnocr/2.2",
        archive: "densenet_lite_136-fc-onnx.zip"
    },
    ModelSpec {
        name: "densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "通用中文识别模型（cnocr 2.3）",
        subfolder: "models/cnocr/2.3",
        archive: "densenet_lite_136-gru-onnx.zip"
    },
    ModelSpec {
        name: "scene-densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "scene-densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "场景文字识别模型（cnocr 2.3）",
        subfolder: "models/cnocr/2.3",
        archive: "scene-densenet_lite_136-gru-onnx.zip"
    },
    ModelSpec {
        name: "doc-densenet_lite_136-gru",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "doc-densenet_lite_136-gru.onnx",
        dict_file: "label_cn_2.3.txt",
        description: "文档识别模型（cnocr 2.3）",
        subfolder: "models/cnocr/2.3",
        archive: "doc-densenet_lite_136-gru-onnx.zip"
    },
    ModelSpec {
        name: "number-densenet_lite_136-fc",
        channels: 1,
        height: 32,
        blank_index: -1,
        onnx_file: "number-densenet_lite_136-fc.onnx",
        dict_file: "label_number.txt",
        description: "数字识别模型（cnocr 2.3）",
        subfolder: "models/cnocr/2.3",
        archive: "number-densenet_lite_136-fc-onnx.zip"
    },
    ModelSpec {
        name: "en_number_mobile_v2.0",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "en_number_mobile_v2.0_rec_infer.onnx",
        dict_file: "en_dict.txt",
        description: "英文数字识别模型（PaddleOCR）",
        subfolder: "models/cnocr/2.3",
        archive: "en_number_mobile_v2.0_rec_infer-onnx.zip"
    },
    ModelSpec {
        name: "chinese_cht_PP-OCRv3",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "chinese_cht_PP-OCRv3_rec_infer.onnx",
        dict_file: "chinese_cht_dict.txt",
        description: "繁体中文识别模型（PaddleOCR PP-OCRv3）",
        subfolder: "models/cnocr/2.3",
        archive: "chinese_cht_PP-OCRv3_rec_infer-onnx.zip"
    },
    ModelSpec {
        name: "ch_PP-OCRv3",
        channels: 3,
        height: 48,
        blank_index: 0,
        onnx_file: "ch_PP-OCRv3_rec_infer.onnx",
        dict_file: "ppocr_keys_v1.txt",
        description: "简体中英文识别模型（PaddleOCR PP-OCRv3）",
        subfolder: "models/cnocr/2.3",
        archive: "ch_PP-OCRv3_rec_infer-onnx.zip"
    },
    ModelSpec {
        name: "ch_ppocr_mobile_v2.0",
        channels: 3,
        height: 32,
        blank_index: 0,
        onnx_file: "ch_ppocr_mobile_v2.0_rec_infer.onnx",
        dict_file: "ppocr_keys_v1.txt",
        description: "简体中英文识别模型（PaddleOCR mobile v2.0）",
        subfolder: "models/cnocr/2.3",
        archive: "ch_ppocr_mobile_v2.0_rec_infer-onnx.zip"
    },
];

/// 按模型名查找注册项
pub fn find_recognizer(name: &str) -> Option<&'static ModelSpec> {
    RECOGNIZERS.iter().find(|spec| spec.name == name)
}

/// PP-OCRv3 文本行检测模型（对应 CnocrCpp 的 `ensure_det_model`）。
/// `channels`/`height`/`blank_index` 对检测没有意义，字典文件为空。
pub const DET_MODEL: ModelSpec = ModelSpec {
    name: "ch_PP-OCRv3_det",
    channels: 3,
    height: 768,
    blank_index: -1,
    onnx_file: "ch_PP-OCRv3_det_infer.onnx",
    dict_file: "",
    description: "文本行检测模型（PP-OCRv3）",
    subfolder: "models/cnstd/1.2",
    archive: "ch_PP-OCRv3_det_infer-onnx.zip",
};

pub fn det_model_spec() -> &'static ModelSpec {
    &DET_MODEL
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
    /// 全部可选模型
    pub const ALL: [UseModel; 9] = [
        UseModel::Cnocr136fc,
        UseModel::DensenetLite136Gru,
        UseModel::SceneDensenet136Gru,
        UseModel::DocDensenet136Gru,
        UseModel::NumberDensenet136Fc,
        UseModel::EnNumber,
        UseModel::ChineseCht,
        UseModel::ChPpOcrV3,
        UseModel::ChPpOcrMobileV2,
    ];

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

    /// 按模型名查找，例如 `"ch_PP-OCRv3"`
    pub fn from_name(name: &str) -> Option<UseModel> {
        UseModel::ALL.into_iter().find(|model| model.name() == name)
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

/// 依次查找 `modle/`、当前目录、缓存目录，返回第一个存在的模型文件
pub fn find_model(spec: &ModelSpec) -> Option<PathBuf> {
    locate(spec.onnx_file, &[PathBuf::from("modle"), PathBuf::new(), cache_dir()])
}

/// 模型文件路径；缺失时 panic 并列出查找过的位置（不联网）
pub fn model_path(spec: &ModelSpec) -> PathBuf {
    find_model(spec).unwrap_or_else(|| panic_missing(spec, "模型可从镜像自动下载，或手动放到上面列出的目录"))
}

/// 模型文件路径；本地缺失且 `auto_download` 为真时，从镜像下载压缩包并解包到缓存目录
pub fn ensure_model(spec: &ModelSpec, auto_download: bool) -> PathBuf {
    if let Some(path) = find_model(spec) {
        return path;
    }
    if spec.archive.is_empty() {
        return panic_missing(spec, "该模型没有可自动下载的压缩包，请手动放置");
    }
    if !auto_download {
        return panic_missing(spec, "已关闭自动下载（Ocr::builder().auto_download(true) 可开启）");
    }

    let cache = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache) {
        panic!("创建模型缓存目录 {} 失败: {}", cache.display(), e);
    }
    let target = cache.join(spec.onnx_file);
    let mut last_error = String::from("没有可用的下载地址");

    for endpoint in download::download_endpoints() {
        let url = download::archive_url(&endpoint, spec.subfolder, spec.archive);
        let zip = cache.join(format!("{}.download", spec.archive));
        eprintln!("下载模型 {}: {}", spec.name, url);
        let result = download::download(&url, &zip).and_then(|()| download::extract_onnx(&zip, &target));
        let _ = std::fs::remove_file(&zip);
        match result {
            Ok(()) => {
                eprintln!("模型已保存到 {}", target.display());
                return target;
            }
            Err(e) => {
                eprintln!("从 {endpoint} 获取模型失败: {e}");
                last_error = e;
            }
        }
    }

    panic!(
        "无法获取模型 {}：{}\n请手动下载后放到 modle/、当前目录或 {}，也可以设 CNOCR_MODEL_ENDPOINT 指向可用的镜像",
        spec.name,
        last_error,
        cache.display()
    )
}

fn panic_missing(spec: &ModelSpec, hint: &str) -> PathBuf {
    panic!(
        "未找到模型文件 {}，查找过 modle/、当前目录和 {}；{}",
        spec.onnx_file,
        cache_dir().display(),
        hint
    )
}

/// 字典文件路径；缺失时 panic 并列出查找过的位置
pub fn dict_path(spec: &ModelSpec) -> PathBuf {
    locate(spec.dict_file, &[PathBuf::from("dict"), PathBuf::new()]).unwrap_or_else(|| {
        panic!("未找到字典文件 {}，请放到 dict/ 或当前目录", spec.dict_file)
    })
}
