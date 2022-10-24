extern crate cc;
use std::path::Path;
fn main() {
    // 以下代码告诉 Cargo ，一旦指定的文件 `src/hello.c` 发生了改变，就重新运行当前的构建脚本
    println!("cargo:rerun-if-changed=src/modle.cpp");
    // 使用 `cc` 来构建一个 C 文件，然后进行静态链接
    cc::Build::new()
        .cpp(true)
        .file("src/modle.cpp")
        .cpp_link_stdlib("stdc++")
        .flag("-L /home/len/Lib/onnxruntime-linux-x64-1.11.1/lib/")
        .flag("-lonnxruntime")
        .include(Path::new("/home/len/Lib/onnxruntime-linux-x64-1.11.1/include/"))
        .include(Path::new("src/modle.h"))
        .compile("modle");
        println!("cargo:rustc-link-search=native={}", "/home/len/Lib/onnxruntime-linux-x64-1.11.1/lib/");
        println!("cargo:rustc-link-lib=onnxruntime");
}