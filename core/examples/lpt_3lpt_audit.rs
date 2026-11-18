//! Phase-A/B audit: 3LPT dependency graph + invariant-basis residual report.
//!
//! ```sh
//! cargo run --release --manifest-path core/Cargo.toml --example lpt_3lpt_audit
//! ```
use hcc_core::ics::PkShape;
use hcc_core::lpt_closure::{
    assess_invariant_basis, cost_model, dependency_nodes, InvariantBasisReport,
};
use hcc_core::Cosmo;
use serde_json::json;
use std::time::Instant;

fn print_graph(dim: usize, max_order: usize) {
    let nodes = dependency_nodes(max_order, dim);
    println!("=== Unmerged term graph dim={dim} order≤{max_order} ({} terms) ===", nodes.len());
    for n in &nodes {
        println!(
            "  [{:>2}] order={} kind={:<6} coef={:+.8} parents={:?}  {}",
            n.index, n.order, n.kind, n.eds_coef, n.parents, n.label
        );
    }
    let cost = cost_model(max_order, dim);
    println!(
        "  cost: unmerged={} merged_eds={} mu2={} mu3={} curl={} approx_ffts={} tensor_fields={}",
        cost.n_terms_unmerged,
        cost.n_terms_merged_eds,
        cost.n_mu2,
        cost.n_mu3,
        cost.n_curl,
        cost.approx_fft_count,
        cost.tensor_field_count
    );
}

fn run_basis(dim: usize, n: usize, seed: u64) -> InvariantBasisReport {
    let mut c = Cosmo::new(dim, n, 1.0);
    c.keep_tensors = Some(true);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
    let t0 = Instant::now();
    c.build_lpt(3);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let report = assess_invariant_basis(c.lpt_ref()).expect("basis assess");
    println!(
        "basis dim={dim} n={n}: build_ms={build_ms:.1} rel_div_exact={:.3e} rel_curl_exact={:.3e} rel_div_m1_local={:.3e} coefs={:?}",
        report.rel_div3_exact_basis,
        report.rel_curl3_exact_basis,
        report.rel_div3_m1_local_fit,
        report.m1_local_fit_coefs
    );
    report
}

fn main() {
    for dim in [2usize, 3] {
        for order in 2..=4 {
            print_graph(dim, order);
            println!();
        }
    }
    let r2 = run_basis(2, 64, 7);
    let r3 = run_basis(3, 24, 7);
    let r3b = run_basis(3, 32, 11);
    let out = json!({
        "graphs": {
            "3d_order3": dependency_nodes(3, 3).iter().map(|n| json!({
                "index": n.index, "order": n.order, "kind": n.kind,
                "edsCoef": n.eds_coef, "parents": n.parents, "label": n.label
            })).collect::<Vec<_>>(),
            "costs": (2..=4).map(|o| {
                let c = cost_model(o, 3);
                json!({"order":o,"unmerged":c.n_terms_unmerged,"approxFfts":c.approx_fft_count,"tensorFields":c.tensor_field_count})
            }).collect::<Vec<_>>()
        },
        "basis": [
            {"dim": r2.dim, "n": r2.n, "relDivExact": r2.rel_div3_exact_basis,
             "relCurlExact": r2.rel_curl3_exact_basis, "relDivM1Local": r2.rel_div3_m1_local_fit,
             "m1Coefs": r2.m1_local_fit_coefs},
            {"dim": r3.dim, "n": r3.n, "relDivExact": r3.rel_div3_exact_basis,
             "relCurlExact": r3.rel_curl3_exact_basis, "relDivM1Local": r3.rel_div3_m1_local_fit,
             "m1Coefs": r3.m1_local_fit_coefs},
            {"dim": r3b.dim, "n": r3b.n, "relDivExact": r3b.rel_div3_exact_basis,
             "relCurlExact": r3b.rel_curl3_exact_basis, "relDivM1Local": r3b.rel_div3_m1_local_fit,
             "m1Coefs": r3b.m1_local_fit_coefs},
        ]
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
