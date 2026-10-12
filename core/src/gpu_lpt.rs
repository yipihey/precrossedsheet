//! Typed boundary for GPU construction of raw nLPT spatial fields.
//! Initial modes, cosmological growth, temporal evaluation and validity checks
//! retain the Rust reference conventions. GPU success never builds CPU LPT.
use crate::{laboratory::{self, LabRequest, LabSession, PendingLabRun}, lpt::Lpt, fft::FftEngine, growth::{term_specs, Parents}, Cosmo};
use serde_json::{json, Value};
use std::{cell::RefCell, sync::{Arc, OnceLock}};

pub struct GpuLptSeed {
    request: LabRequest,
    initial: RefCell<Option<Cosmo>>,
    reference: OnceLock<Vec<f32>>,
}
impl GpuLptSeed {
    pub fn new(r: &LabRequest) -> Result<Self, String> {
        r.validate()?;
        if ![2,3].contains(&r.dim) || !r.n.is_power_of_two() {
            return Err("GPU LPT requires a radix-2 2D or 3D Lagrangian lattice".into());
        }
        Ok(Self { request:r.clone(), initial:RefCell::new(Some(laboratory::initial_cosmo(r))), reference:OnceLock::new() })
    }
    pub fn metadata(&self) -> Value {
        let specs = term_specs(4, self.request.dim);
        let field_stride = self.request.dim + self.request.dim * self.request.dim + 3;
        let particles = self.request.n.pow(self.request.dim as u32);
        let term_scalars = particles * field_stride;
        let rows: Vec<Value> = specs.iter().map(|s| {
            let (kind,a,b,c) = match s.parents { Parents::Linear => (0,0,0,0), Parents::Mu2(a,b)=>(1,a,b,0), Parents::Mu3(a,b,c)=>(2,a,b,c), Parents::Curl(a,b)=>(3,a,b,0) };
            json!({"order":s.order,"kind":kind,"a":a,"b":b,"c":c})
        }).collect();
        json!({"dim":self.request.dim,"n":self.request.n,"order":4,"termCount":specs.len(),"specs":rows,
            "fieldStride":field_stride,"termScalars":term_scalars,"totalScalars":specs.len()*term_scalars,
            "request":self.request,"inputPrecision":"f64 initial modes cast to complex f32; no renormalization","growthPrecision":"Rust f64"})
    }
    pub fn delta_hat(&self) -> Result<Vec<f32>, String> {
        let state = self.initial.borrow();
        let c = state.as_ref().ok_or("GPU LPT seed has already been consumed")?;
        Ok(c.delta0_hat.iter().flat_map(|v| [v.re as f32,v.im as f32]).collect())
    }
    /// Independent CPU raw-term oracle, evaluated only when explicitly requested.
    pub fn reference_fields(&self) -> Result<&[f32], String> {
        if let Some(v) = self.reference.get() { return Ok(v); }
        let state = self.initial.borrow();
        let c = state.as_ref().ok_or("GPU LPT seed has already been consumed")?;
        let mut engine = FftEngine::new(&c.grid);
        let lpt = Lpt::new_unmerged(&c.grid,&mut engine,&c.delta0_hat,4,c.cosmo,true);
        let _ = self.reference.set(lpt.packed_fields());
        Ok(self.reference.get().unwrap())
    }
    fn complete_initial(&self, fields: &[f32]) -> Result<Arc<Cosmo>, String> {
        let mut state = self.initial.borrow_mut();
        let c = state.as_ref().ok_or("GPU LPT seed has already been consumed")?;
        let lpt = Lpt::from_packed_fields(&c.grid,4,c.cosmo,fields)?;
        let mut c = state.take().unwrap();
        c.lpt = Some(lpt);
        Ok(Arc::new(c))
    }
    pub fn finish_gpu(&self, fields: &[f32]) -> Result<PendingLabRun, String> {
        let c = self.complete_initial(fields)?;
        laboratory::prepare_pending_from_initial_source(&self.request,&c,"gpu-f32")
    }
    pub fn finish_gpu_with_session(&self, session: &mut LabSession, fields: &[f32]) -> Result<PendingLabRun, String> {
        let c = self.complete_initial(fields)?;
        session.adopt_initial(&self.request,c,"gpu-f32")
    }
}
