use cnocr_rs::ocr::{Ocr,OcrTraitConst};
use std::time::Instant;
fn main() {
    let t1=Instant::now();
    let ocr=Ocr::new();
    println!("载入用时:{}ms",t1.elapsed().as_millis());
    let t2=Instant::now();
    let vs=ocr.from_path("cnocr.png".to_string());
    for i in vs{
        println!("识别到:{:?}",i);
    }
    println!("识别用时:{}ms",t2.elapsed().as_millis());
    println!("总用时用时:{}ms",t1.elapsed().as_millis());
}
