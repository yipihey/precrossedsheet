//! Repeatable native stage profile for the laboratory preparation boundary.
//!
//! Run with `cargo run --release --manifest-path core/Cargo.toml --example lab_profile -- 3`.
use hcc_core::laboratory::{prepare, LabRequest};
use serde_json::{json, Value};
use std::{env, time::Instant};

fn request(dim: usize, n: usize) -> LabRequest {
    LabRequest {
        dim,
        n,
        mesh: n,
        force_mesh: n,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    }
}

fn elapsed_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_secs_f64() * 1000.0)
}

fn stats(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    json!({"medianMs":median,"minMs":sorted[0],"maxMs":sorted[sorted.len()-1],"samplesMs":samples})
}

fn run_case(label: &str, dim: usize, n: usize, repeats: usize) -> Result<Value, String> {
    let request = request(dim, n);
    // Warm all stages once. Warm-up is intentionally excluded from samples.
    let warm = prepare(&request)?;
    let cpu = warm.direct_cpu();
    let _quantized = warm.direct_quantized_cpu();
    let _fd = warm.finite_difference_cpu();
    let result = warm.finish(&cpu, r#"{"backend":"cpu"}"#)?;
    let _serialized = result.to_string();

    let mut prepare_ms = Vec::with_capacity(repeats);
    let mut direct_ms = Vec::with_capacity(repeats);
    let mut reference_check_ms = Vec::with_capacity(repeats);
    let mut fd_ms = Vec::with_capacity(repeats);
    let mut finish_ms = Vec::with_capacity(repeats);
    let mut serialize_ms = Vec::with_capacity(repeats);
    let mut total_ms = Vec::with_capacity(repeats);
    let mut output_bytes = 0;

    for _ in 0..repeats {
        let (prepared, t) = elapsed_ms(|| prepare(&request));
        let prepared = prepared?;
        prepare_ms.push(t);
        let (channels, t) = elapsed_ms(|| prepared.direct_cpu());
        direct_ms.push(t);
        let (quantized_channels, t) = elapsed_ms(|| prepared.direct_quantized_cpu());
        reference_check_ms.push(t);
        let (fd, t) = elapsed_ms(|| prepared.finite_difference_cpu().to_vec());
        fd_ms.push(t);
        let (result, t) = elapsed_ms(|| prepared.finish(&channels, r#"{"backend":"cpu"}"#));
        let result = result?;
        finish_ms.push(t);
        let (serialized, t) = elapsed_ms(|| result.to_string());
        output_bytes = serialized.len();
        serialize_ms.push(t);
        total_ms.push(0.0);
        std::hint::black_box((quantized_channels, fd, serialized));
    }

    // The normal CPU run excludes the optional packed-input reference check.
    for i in 0..repeats {
        total_ms[i] = prepare_ms[i] + direct_ms[i] + fd_ms[i] + finish_ms[i] + serialize_ms[i];
    }

    Ok(json!({
        "case":label,
        "request":serde_json::to_value(&request).expect("serializing a laboratory request cannot fail"),
        "modeCount":(request.n / 4).min(6),
        "estimatedPeakBytes":request.estimated_peak_bytes(),
        "outputBytes":output_bytes,
        "repeats":repeats,
        "stages":{
            "prepare":stats(&prepare_ms),
            "directCpuF64":stats(&direct_ms),
            "packedInputReferenceCheck":stats(&reference_check_ms),
            "finiteDifferenceCpuF64":stats(&fd_ms),
            "finish":stats(&finish_ms),
            "serialize":stats(&serialize_ms),
            "cpuRunTotalExcludingReferenceCheck":stats(&total_ms)
        }
    }))
}

fn main() {
    let repeats = env::args().nth(1).map(|s| s.parse::<usize>().expect("repeat count must be an integer")).unwrap_or(3);
    let dim = env::args().nth(2).map(|s| s.parse::<usize>().expect("dimension must be an integer")).unwrap_or(2);
    assert!(repeats > 0, "repeat count must be positive");
    assert!((1..=3).contains(&dim), "dimension must be 1, 2 or 3");
    let profile = json!({
        "profile":"prepared-lab-rust-v1",
        "warmupRuns":1,
        "clock":"std::time::Instant; native process, release profile recommended",
        "cases":[
            run_case("small",dim,16,repeats).expect("small profile request failed"),
            run_case("32",dim,32,repeats).expect("32 profile request failed"),
            run_case("64",dim,64,repeats).expect("64 profile request failed")
        ]
    });
    println!("{}", profile);
}
