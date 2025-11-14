//! Build script to generate C header file

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");

    // Generate C header
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("LGI_H")
        .with_documentation(true)
        .with_pragma_once(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(output_dir.join("lgi.h"));

    println!("cargo:rerun-if-changed=src/lib.rs");
}
