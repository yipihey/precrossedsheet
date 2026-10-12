// CAP40 graph-based tetrahedron clipper adapted from hopf-cole-cosmology exact3d.js.
// Upstream repository pinned at 3a3a13b9ba817c5b4f8f9f26314715845e13d4ad;
// source clipper commit: 5ae0550eaea89c85781de8cebb78462cb50aca6c (MIT; see UPSTREAM-MIT-NOTICE.txt).
// Production wrapper adds explicit link-walk diagnostics before accepting any density result.
export const WGSL_CLIP3D = /* wgsl */`
const CAP: u32 = 40u;
const NIL: u32 = 255u;
struct Poly { pos: array<vec3<f32>, CAP>, phi: array<f32, CAP>, nbr: array<u32, CAP>, n: u32 };   // phi: scalar attribute per vertex (linear inside)

var<private> P_T: Poly;      // the tetrahedron
var<private> P_X: Poly;      // ... clipped to the x-slab
var<private> P_Y: Poly;      // ... and the y-slab
var<private> P_A: Poly;      // scratch (second half of every slab clip)
var<private> P_B: Poly;      // the cell piece
var<private> sdv: array<f32, CAP>;
var<private> idxv: array<u32, CAP>;
var<private> emk: array<u32, CAP>;
var<private> g_ovf: u32;

fn pack3(a: u32, b: u32, c: u32) -> u32 { return a | (b << 8u) | (c << 16u); }
fn nbget(p: ptr<private, Poly>, v: u32, k: u32) -> u32 { return ((*p).nbr[v] >> (8u * k)) & 255u; }

fn init_tet(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, f: vec4<f32>) {   // positively oriented: (b-a, c-a, d-a) right handed
  P_T.pos[0] = a; P_T.pos[1] = b; P_T.pos[2] = c; P_T.pos[3] = d;
  P_T.phi[0] = f.x; P_T.phi[1] = f.y; P_T.phi[2] = f.z; P_T.phi[3] = f.w;
  P_T.nbr[0] = pack3(1u, 3u, 2u); P_T.nbr[1] = pack3(2u, 3u, 0u); P_T.nbr[2] = pack3(0u, 3u, 1u); P_T.nbr[3] = pack3(1u, 2u, 0u);
  P_T.n = 4u;
}

// One-sided clip of src into dst: keep sgn * (x[ax] - coord) <= 0.  Returns 0 (empty), 1 (nothing clipped, dst = copy of src) or 2.
fn clip(src: ptr<private, Poly>, dst: ptr<private, Poly>, ax: u32, coord: f32, sgn: f32) -> u32 {
  let onv = (*src).n;
  var nin = 0u;
  for (var v = 0u; v < onv; v++) {
    let s = sgn * ((*src).pos[v][ax] - coord);
    sdv[v] = s;
    if (s > 0.0) { idxv[v] = NIL; } else { idxv[v] = nin; nin++; }
  }
  if (nin == 0u) { return 0u; }
  if (nin == onv) {
    for (var v = 0u; v < onv; v++) { (*dst).pos[v] = (*src).pos[v]; (*dst).phi[v] = (*src).phi[v]; (*dst).nbr[v] = (*src).nbr[v]; }
    (*dst).n = onv;
    return 1u;
  }
  var nn = nin;
  for (var v = 0u; v < onv; v++) {
    let iv = idxv[v];
    if (iv == NIL) { continue; }
    var packed = 0u;
    for (var np = 0u; np < 3u; np++) {
      let w = nbget(src, v, np);
      var tgt = idxv[w];
      if (tgt == NIL) {
        if (nn >= CAP) { g_ovf = 1u; return 0u; }
        let wa = -sdv[w];
        let wb = sdv[v];
        let inv = 1.0 / (wa + wb);
        (*dst).pos[nn] = (wa * (*src).pos[v] + wb * (*src).pos[w]) * inv;
        (*dst).phi[nn] = (wa * (*src).phi[v] + wb * (*src).phi[w]) * inv;
        (*dst).nbr[nn] = pack3(iv, NIL, NIL);
        tgt = nn; nn++;
      }
      packed = packed | (tgt << (8u * np));
    }
    (*dst).pos[iv] = (*src).pos[v];
    (*dst).phi[iv] = (*src).phi[v];
    (*dst).nbr[iv] = packed;
  }
  // link the new vertices around the cut face: walk (np + 1) % 3 from each until the next new vertex
  for (var vs = nin; vs < nn; vs++) {
    var vc = vs;
    var vn = nbget(dst, vc, 0u);
    var linked = false;
    for (var it = 0u; it < 64u; it++) {
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(dst, vn, k) == vc) { np = k; break; } }
      vc = vn;
      vn = nbget(dst, vc, (np + 1u) % 3u);
      if (vc >= nin) { linked = true; break; }
    }
    if (!linked) { g_ovf = 2u; return 0u; }
    (*dst).nbr[vs] = ((*dst).nbr[vs] & ~(255u << 16u)) | (vc << 16u);
    (*dst).nbr[vc] = ((*dst).nbr[vc] & ~(255u << 8u)) | (vs << 8u);
  }
  (*dst).n = nn;
  return 2u;
}

// Clip to lo <= x[ax] <= hi (through the scratch polytope P_A).
fn slab(src: ptr<private, Poly>, dst: ptr<private, Poly>, ax: u32, lo: f32, hi: f32) -> u32 {
  if (clip(src, &P_A, ax, lo, -1.0) == 0u) { return 0u; }
  return clip(&P_A, dst, ax, hi, 1.0);
}

// (volume, first moments) about 'org' by fanning every face (Koehl recursion, order 1)
fn moments1(p: ptr<private, Poly>, org: vec3<f32>) -> vec4<f32> {
  let nv = (*p).n;
  for (var v = 0u; v < nv; v++) { emk[v] = 0u; }
  // Fan each oriented face about a point inside this convex clipped piece.
  // Using the cell corner can put the fan origin outside a sliver, causing
  // large positive/negative signed tetrahedra to cancel to a negative f32
  // volume. The vertex average lies inside every convex polytope here.
  var center = vec3<f32>(0.0);
  for (var v = 0u; v < nv; v++) { center += (*p).pos[v] / f32(nv); }
  var m0 = 0.0;
  var m1 = vec3<f32>(0.0);
  for (var vstart = 0u; vstart < nv; vstart++) {
    for (var pstart = 0u; pstart < 3u; pstart++) {
      if ((emk[vstart] & (1u << pstart)) != 0u) { continue; }
      var vcur = vstart;
      var pnext = pstart;
      emk[vcur] = emk[vcur] | (1u << pnext);
      var vnext = nbget(p, vcur, pnext);
      let v0 = (*p).pos[vcur] - center;
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
      vcur = vnext;
      pnext = (np + 1u) % 3u;
      emk[vcur] = emk[vcur] | (1u << pnext);
      vnext = nbget(p, vcur, pnext);
      var closed = false;
      for (var it = 0u; it < 64u; it++) {
        if (vnext == vstart) { closed = true; break; }
        let v2 = (*p).pos[vcur] - center;
        let v1 = (*p).pos[vnext] - center;
        let sixv = dot(v0, cross(v1, v2));
        m0 += sixv;
        m1 += sixv * (v0 + v1 + v2);
        np = 0u;
        for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
        vcur = vnext;
        pnext = (np + 1u) % 3u;
        emk[vcur] = emk[vcur] | (1u << pnext);
        vnext = nbget(p, vcur, pnext);
      }
      if (!closed) { g_ovf = 2u; return vec4<f32>(0.0); }
    }
  }
  let volume = m0 / 6.0;
  // The accumulated first moment is about center. Restore the existing API's
  // moment-about-org convention without changing the volume result.
  return vec4<f32>(volume, m1 / 24.0 + (center - org) * volume);
}


// (volume, integral of the linear scalar phi) of the polytope, fanned from its vertex 0 so that phi is only interpolated, never
// extrapolated (the P1 shape of a sliver tetrahedron has large barycentric gradients: evaluating it at a distant origin would cancel)
fn integ(p: ptr<private, Poly>) -> vec2<f32> {
  let nv = (*p).n;
  for (var v = 0u; v < nv; v++) { emk[v] = 0u; }
  let o = (*p).pos[0];
  let f0 = (*p).phi[0];
  var vol = 0.0;
  var ip = 0.0;
  for (var vstart = 0u; vstart < nv; vstart++) {
    for (var pstart = 0u; pstart < 3u; pstart++) {
      if ((emk[vstart] & (1u << pstart)) != 0u) { continue; }
      var vcur = vstart;
      var pnext = pstart;
      emk[vcur] = emk[vcur] | (1u << pnext);
      var vnext = nbget(p, vcur, pnext);
      let v0 = (*p).pos[vstart] - o;
      let g0 = (*p).phi[vstart];
      var np = 0u;
      for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
      vcur = vnext;
      pnext = (np + 1u) % 3u;
      emk[vcur] = emk[vcur] | (1u << pnext);
      vnext = nbget(p, vcur, pnext);
      var closed = false;
      for (var it = 0u; it < 64u; it++) {
        if (vnext == vstart) { closed = true; break; }
        let v2 = (*p).pos[vcur] - o;
        let v1 = (*p).pos[vnext] - o;
        let sixv = dot(v0, cross(v1, v2));
        vol += sixv;
        ip += sixv * (f0 + g0 + (*p).phi[vcur] + (*p).phi[vnext]);
        np = 0u;
        for (var k = 0u; k < 3u; k++) { if (nbget(p, vnext, k) == vcur) { np = k; break; } }
        vcur = vnext;
        pnext = (np + 1u) % 3u;
        emk[vcur] = emk[vcur] | (1u << pnext);
        vnext = nbget(p, vcur, pnext);
      }
      if (!closed) { g_ovf = 2u; return vec2<f32>(0.0); }
    }
  }
  return vec2<f32>(vol / 6.0, ip / 24.0);
}

// Visit the cells [blo, bhi) (relative to the base cell) overlapped by the tetrahedron P_T and emit their moments.
fn walk_tet(blo: vec3<i32>, bhi: vec3<i32>) {
  for (var ci = blo.x; ci < bhi.x; ci++) {
    if (slab(&P_T, &P_X, 0u, f32(ci), f32(ci + 1)) == 0u) { continue; }
    var ylo = P_X.pos[0].y; var yhi = ylo;
    for (var v = 1u; v < P_X.n; v++) { ylo = min(ylo, P_X.pos[v].y); yhi = max(yhi, P_X.pos[v].y); }
    let j0 = max(blo.y, i32(floor(ylo)));
    let j1 = min(bhi.y, i32(ceil(yhi)));
    for (var cj = j0; cj < j1; cj++) {
      if (slab(&P_X, &P_Y, 1u, f32(cj), f32(cj + 1)) == 0u) { continue; }
      var zlo = P_Y.pos[0].z; var zhi = zlo;
      for (var v = 1u; v < P_Y.n; v++) { zlo = min(zlo, P_Y.pos[v].z); zhi = max(zhi, P_Y.pos[v].z); }
      let k0 = max(blo.z, i32(floor(zlo)));
      let k1 = min(bhi.z, i32(ceil(zhi)));
      for (var ck = k0; ck < k1; ck++) {
        if (slab(&P_Y, &P_B, 2u, f32(ck), f32(ck + 1)) == 0u) { continue; }
        emit(vec3<i32>(ci, cj, ck), &P_B);
      }
    }
  }
}
`;
