# RustCnOcr

- 一份来自python cnocr的Rust实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- onnxruntime 自 https://github.com/microsoft/onnxruntime/releases 下载后 自行更改 build.rs

## 待解决问题
- onnxruntime 库的原因，暂时仍使用c++代码解决onnxruntime库调用的问题
- 切换ocr识别语言包

## 使用演示

```rust
use RustCnOcr::Ocr::{ocr,OcrTraitConst};
fn main() {
    let ocr=ocr::new();
    let vs=ocr.from_path("cnocr.png".to_string());
    for i in vs{
        println!("识别到:{:?}",i);
    }
}

```
```sh
识别到:("cnocr自V2.1.2之后，可直接使用的模型包含两类：1）cnocr自己训练的模型，通常会包含PyTorch和", 0.38941005)
识别到:("ONNX版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于cnocr中。", 0.42324248)
```