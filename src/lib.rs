//! 纯 Rust 的中文 OCR，推理走 `ort`（crates.io 上的 ONNX Runtime 绑定），图像处理只用 `image`。
//!
//! - [`ocr`]：识别器与文字行切分
//! - [`cnstd`]：PP-OCRv3 文本行检测（DBNet）
//! - [`geometry`]：文本框几何计算
//! - [`model_registry`]：模型清单与本地文件定位
//! - [`download`]：模型自动下载与 zip 解包
//!
//! 用法见 `src/main.rs`。

pub mod cnstd;
pub mod download;
pub mod geometry;
pub mod model_registry;
pub mod ocr;
