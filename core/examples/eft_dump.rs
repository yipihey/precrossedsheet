use hcc_core::{eft::run_dual_smoothing_counterterms, laboratory::LabRequest};
fn main() {
    for &uv in &[4.0_f64, 5.0, 6.0, 8.0] {
        let base = LabRequest {
            dim: 3, n: 32, mesh: 64, force_mesh: 32,
            ic: "gaussian".into(), seed: 7, box_size: 300.,
            initial_cutoff: 4, epoch: "scaleFactor".into(), scale_factor: 1.0,
            fraction: 1.0, order: 4, steps: 32, smoothing: 0.0,
            transfer: "smooth".into(), estimators: vec![],
            extended: true, // allow if needed
            ..Default::default()
        };
        // n=32 is within default envelope
        let base = LabRequest { extended: false, ..base };
        match run_dual_smoothing_counterterms(&base, uv, None, 0.08) {
            Ok(out) => {
                let o4 = &out["orders"][3];
                let k = o4["ir"]["k"].as_array().unwrap();
                let puv = o4["uv"]["power"].as_array().unwrap(); // may be different k; use delta
                let pir = o4["ir"]["power"].as_array().unwrap();
                let dpow = o4["deltaPower"].as_array().unwrap();
                let pred = o4["irPlusCounterterm"].as_array().unwrap();
                // relative residual of counterterm vs UV on first fit bins
                let nfit = o4["fit"]["nBins"].as_u64().unwrap() as usize;
                let mut num = 0.0; let mut den = 0.0;
                for i in 0..nfit.min(k.len()) {
                    let uv_i = pir[i].as_f64().unwrap() + dpow[i].as_f64().unwrap();
                    let err = uv_i - pred[i].as_f64().unwrap();
                    num += err*err; den += uv_i*uv_i;
                    let _ = puv;
                }
                println!(
                    "RfUV={uv:.1} RfIR={:.2} σUV={:.3} σIR={:.3} invUV={:.4} invIR={:.3e} cs2={:.1} relRMS={:.3e}",
                    out["filters"]["irSmoothing"].as_f64().unwrap(),
                    out["filters"]["sigmaLinUv"].as_f64().unwrap(),
                    out["filters"]["sigmaLinIr"].as_f64().unwrap(),
                    o4["uv"]["invertedElementFraction"].as_f64().unwrap(),
                    o4["ir"]["invertedElementFraction"].as_f64().unwrap(),
                    o4["fit"]["cs2"].as_f64().unwrap(),
                    (num/den.max(1e-300)).sqrt()
                );
            }
            Err(e) => println!("RfUV={uv}: {e}"),
        }
    }
}
