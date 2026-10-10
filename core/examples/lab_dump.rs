//! Native request/response runner for reproducible figures and integration checks.
use std::io::Read;
fn main() {
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request).unwrap();
    println!("{}", hcc_core::laboratory::run_json(&request));
}
