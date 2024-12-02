use std::{fs::File, io::Write};

fn main() {
    let bindings = bindgen::Builder::default()
        .header("./cuda-build/wrapper.h")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .size_t_is_usize(true)
        .allowlist_type("^cublas.*")
        .allowlist_function("^cublas.*")
        .allowlist_type("^cuda.*")
        .allowlist_function("^cuda.*")
        .allowlist_type("^cudnn.*")
        .allowlist_function("^cudnn.*")
        .layout_tests(false)
        .clang_arg("-I/usr/include/")
        .clang_arg("-I/usr/local/cuda/include")
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings");
    let mut writer = File::create("./cuda-bindings/src/bindings.rs").unwrap();
    writer
        .write_all("#![allow(warnings)]\n".as_bytes())
        .unwrap();
    bindings
        .write(Box::new(writer))
        .expect("Couldn't write bindings!");
}
