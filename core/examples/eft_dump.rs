use hcc_core::{eft::run_dual_smoothing_counterterms, laboratory::LabRequest};
fn main() {
    let base = LabRequest {
        dim: 3, n: 32, mesh: 64, force_mesh: 32,
        ic: "gaussian".into(), seed: 7, box_size: 300.,
        initial_cutoff: 4, epoch: "scaleFactor".into(), scale_factor: 1.0,
        fraction: 1.0, order: 4, steps: 32, smoothing: 0.0,
        transfer: "smooth".into(), estimators: vec![],
        ..Default::default()
    };
    let out = run_dual_smoothing_counterterms(&base, 5.0, None, 0.08).unwrap();
    println!("filters={}", out["filters"]);
    println!("convention={}", out["convention"]);
    for ord in &[1usize, 4] {
        let o = &out["orders"][ord - 1];
        let fit_pk = &o["fit"];
        let fit_d = &o["displacement"]["fit"];
        let stoch = &o["displacement"]["stochastic"];
        println!(
            "\n=== {ord}LPT  P(k) cs2={:.3}  displacement cs2={:.3} ===",
            fit_pk["cs2"].as_f64().unwrap_or(f64::NAN),
            fit_d["cs2"].as_f64().unwrap_or(f64::NAN),
        );
        println!(
            "  vertex RMS before/after CT: {:.4e} → {:.4e}  (explained {:.1}%, corr {:.3})",
            fit_d["rmsBefore"].as_f64().unwrap_or(f64::NAN),
            fit_d["rmsAfter"].as_f64().unwrap_or(f64::NAN),
            100.0 * fit_d["varianceExplained"].as_f64().unwrap_or(0.0),
            fit_d["correlation"].as_f64().unwrap_or(f64::NAN),
        );
        println!(
            "  stochastic ensemble: {} draws, coverage within 1σ of CT+noise = {:.3}",
            stoch["nRealizations"].as_u64().unwrap_or(0),
            stoch["ensembleCoverageWithin1Sigma"].as_f64().unwrap_or(f64::NAN),
        );
        let k = o["ir"]["k"].as_array().unwrap();
        let pir = o["ir"]["power"].as_array().unwrap();
        let dpow = o["deltaPower"].as_array().unwrap();
        let pred = o["irPlusCounterterm"].as_array().unwrap();
        println!("k        P_IR       P_UV       UV/IR    ΔP/P_IR   (IR+ctr)/IR");
        for i in 0..k.len().min(8) {
            let ki = k[i].as_f64().unwrap();
            let ir = pir[i].as_f64().unwrap();
            let dp = dpow[i].as_f64().unwrap();
            let uv = ir + dp;
            let ctr = pred[i].as_f64().unwrap();
            println!(
                "{ki:.4}  {ir:.4e}  {uv:.4e}  {r:.5}  {drel:.5}  {crel:.5}",
                r = uv / ir,
                drel = dp / ir,
                crel = ctr / ir
            );
        }
    }
}
