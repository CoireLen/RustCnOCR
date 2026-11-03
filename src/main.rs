use cnocr_rs::ocr::{Ocr, OcrTraitConst};
use std::time::Instant;

fn main() {
    // 用法: cnocr-rs [图片路径] [模型名] [-d] [--no-download]
    // -d 表示先用 PP-OCRv3 检测模型切分文本行；--no-download 关闭模型自动下载
    let mut detection = false;
    let mut auto_download = true;
    let mut positional: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "-d" || arg == "--detection" {
            detection = true;
        } else if arg == "--no-download" {
            auto_download = false;
        } else if !arg.is_empty() {
            positional.push(arg);
        }
    }
    let path = positional.first().map(String::as_str).unwrap_or("cnocr.png");

    let t1 = Instant::now();
    let mut builder = Ocr::builder().detection(detection).auto_download(auto_download);
    if let Some(name) = positional.get(1) {
        builder = builder.model_name(name);
    }
    let ocr = builder.build();
    println!("模型:{} 文本行检测:{} 载入用时:{}ms", ocr.model().name, ocr.uses_detection(), t1.elapsed().as_millis());

    let t2 = Instant::now();
    for line in ocr.from_path(path) {
        println!("识别到:{:?}", line);
    }
    println!("识别用时:{}ms", t2.elapsed().as_millis());
    println!("总用时:{}ms", t1.elapsed().as_millis());
}
