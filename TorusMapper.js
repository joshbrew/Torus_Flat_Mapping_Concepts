
/* =========================================================================
   Constants and parametric surface functions
   - stdPos: standard torus parameterization (u,v angles in radians)
   - cliffordPos: Clifford torus stereographic mapping for alternate appearance
   - NOTE: display ordering swaps y and z to match original layout
   ========================================================================= */

const TWO_PI = Math.PI*2;

/**
 * Standard torus parameterization
 * u - angle around main hole (radians)
 * v - angle around tube (radians)
 * Rv - major radius
 * rv - minor radius
 * returns THREE.Vector3 in display coordinate ordering (x, z, y) to match original visuals
 */
function stdPos(u,v,Rv,rv){
  const cu=Math.cos(u), su=Math.sin(u), cv=Math.cos(v), sv=Math.sin(v);
  const x=(Rv+rv*cv)*cu;
  const y=(Rv+rv*cv)*su;
  const z=rv*sv;
  // return new Vector3(x, z, y) so the scene's up/forward remain consistent with original
  return new THREE.Vector3(x, z, y);
}

/**
 * Clifford torus stereographic mapping
 * used for the alternative 'clifford' mode
 * The scale parameter adjusts stereographic projection scale
 */
function cliffordPos(u,v,scale=1.8){
  const s = Math.SQRT1_2;
  const x1 = s * Math.cos(u);
  const y1 = s * Math.sin(u);
  const z1 = s * Math.cos(v);
  const w1 = s * Math.sin(v);
  const d = 1 - w1;
  return new THREE.Vector3((x1 / d) * scale, (z1 / d) * scale, (y1 / d) * scale);
}

/* =========================================================================
   small linear algebra helper - 3x3 solver
   - used by cross-square Kasa circle fit
   ========================================================================= */

/**
 * Solve a 3x3 linear system A x = b using Gaussian elimination
 * A - 3x3 array-of-arrays
 * b - length-3 array
 * returns array [x0,x1,x2] or null on singular matrix
 */
function solve3x3(A, b){
  const M = [
    [A[0][0],A[0][1],A[0][2], b[0]],
    [A[1][0],A[1][1],A[1][2], b[1]],
    [A[2][0],A[2][1],A[2][2], b[2]]
  ];
  for(let i=0;i<3;i++){
    let piv=i;
    for(let j=i+1;j<3;j++) if(Math.abs(M[j][i])>Math.abs(M[piv][i])) piv=j;
    if(Math.abs(M[piv][i])<1e-12) return null;
    if(piv!==i){ const tmp=M[i]; M[i]=M[piv]; M[piv]=tmp; }
    const d = M[i][i];
    for(let k=i;k<4;k++) M[i][k]/=d;
    for(let j=0;j<3;j++) if(j!==i){
      const f=M[j][i];
      if(Math.abs(f) < 1e-18) continue;
      for(let k=i;k<4;k++) M[j][k] -= f*M[i][k];
    }
  }
  return [M[0][3], M[1][3], M[2][3]];
}

/* =========================================================================
   TorusMapper class
   - encapsulates mapping logic from fractional UV (0..1) to 3D
   - provides helpers to sample the rectangle perimeter, spiral arc, cross-square
   - outputs are plain JS arrays suitable for building line geometry
   ========================================================================= */

/**
 * TorusMapper
 * @class
 * @param {Object} params - initial parameters
 * @param {'std'|'clifford'} params.mode - mapping mode
 * @param {number} params.R - major radius (for std mode)
 * @param {number} params.r - minor radius (for std mode)
 * @param {number} params.shear - shear factor: u <- u + shear * v (for spiral mapping)
 */
class TorusMapper {
  constructor(params = {}) {
    // default parameters
    this.params = Object.assign({
      mode: 'std',
      R: 1.2,
      r: 0.44,
      shear: 0.0
    }, params);
    this.TWO_PI = Math.PI * 2;
  }

  /**
   * Update parameters in place
   * @param {Object} newParams - partial params to merge
   * @returns {TorusMapper} this
   */
  updateParams(newParams = {}){ Object.assign(this.params, newParams); return this; }

  /**
   * Set a single parameter
   * @param {string} k - key
   * @param {*} v - value
   * @returns {TorusMapper} this
   */
  setParam(k,v){ this.params[k]=v; return this; }

  /**
   * Return a shallow copy of current params
   * @returns {Object}
   */
  getParams(){ return Object.assign({}, this.params); }

  /**
   * Internal: wrap fractional value into [0,1)
   * @param {number} x
   * @returns {number}
   */
  _wrap01(x){ x%=1; return x<0?x+1:x; }

  /**
   * Internal: get position function based on mode
   * @returns {Function} function(uRadians,vRadians) -> THREE.Vector3
   */
  _getPosFunc(){
    if(this.params.mode === 'clifford') return (u,v)=>cliffordPos(u,v,1.8);
    return (u,v)=>stdPos(u,v,this.params.R,this.params.r);
  }

  /**
   * Map fractional UV in [0,1) to 3D position in world/local torus space
   * - applies shear: u' = u + shear * v
   * @param {number} U - fractional U in [0,1)
   * @param {number} V - fractional V in [0,1)
   * @returns {THREE.Vector3}
   */
  posFromUV(U,V){
    const shear = this.params.shear || 0.0;
    const u = this._wrap01(U) * this.TWO_PI;
    const v = this._wrap01(V) * this.TWO_PI;
    const getPos = this._getPosFunc();
    return getPos(u + shear * v, v);
  }

  /**
   * Compute local frame at fractional UV
   * - numeric finite differences used to estimate tangent directions
   * @param {number} U fractional u
   * @param {number} V fractional v
   * @param {number} [eps=1e-4] small step for finite difference
   * @returns {{p:THREE.Vector3, tu:THREE.Vector3, tv:THREE.Vector3, n:THREE.Vector3}}
   */
  frameAt(U,V,eps=1e-4){
    const P = this.posFromUV(U,V);
    const Pu = this.posFromUV(U+eps,V);
    const Pv = this.posFromUV(U,V+eps);
    const tu = Pu.clone().sub(P).normalize();
    const tv = Pv.clone().sub(P).normalize();
    const n = tu.clone().cross(tv).normalize();
    return { p:P, tu, tv, n };
  }

  /**
   * Compute per-row U counts (Nu) given a base segU and segV.
   * - This uses the standard-torus circumference at each V to scale segU.
   * - options:
   *    equalQuads: boolean -> scale using radial circumference (approx equal arc length around hole)
   *    clampMin: min count per row (default 8)
   *    clampMax: max count per row (default segU*3)
   *    sampleV: number of v samples used when measuring circumference (defaults to segV)
   * - returns Uint16Array length segV with counts per row
   */
  rowCounts(segUBase, segV, opts = {}){
    const opt = Object.assign({ equalQuads: false, clampMin:8, clampMax: segUBase*3, sampleV: segV }, opts);
    const Nu = new Uint16Array(segV);
    // For the standard torus the circumference at tube-angle v is: 2π * (R + r*cos(v))
    // coefficient relative to base 2π*R is (R + r*cos(v)) / R
    // Use that to scale segUBase. If equalQuads requested, we can use the same coefficient logic.
    for(let j=0;j<segV;j++){
      const v = (j / segV) * this.TWO_PI;
      let coef = 1.0;
      if(this.params.mode === 'std'){
        const R = this.params.R, r = this.params.r;
        coef = (R + r * Math.cos(v)) / Math.max(1e-9, R);
      } else {
        // fallback: for non-std modes use uniform
        coef = 1.0;
      }
      let n = Math.round(segUBase * coef);
      n = Math.max(opt.clampMin, Math.min(opt.clampMax, n));
      Nu[j] = n;
    }
    return Nu;
  }

  /**
   * Build the rectangle perimeter mapped to plane and torus
   * - returns uv list, planePts (3D coords on plane), torusPts (3D coords on torus)
   * - also splits the continuous perimeter into loops where wrapping occurs
   *
   * @param {number} Uc center U (fractional)
   * @param {number} Vc center V (fractional)
   * @param {number} Usz size in U (fractional, 0..1)
   * @param {number} Vsz size in V (fractional, 0..1)
   * @param {number} [edge=128] number of samples per edge
   * @returns {{
   *   uv:Array, planePts:Array, torusPts:Array,
   *   planeLoops:Array<Array>, torusLoops:Array<Array>
   * }}
   */
  rectPerimeterData(Uc,Vc,Usz,Vsz, edge=128){
    const Umin = Uc - Usz/2, Umax = Uc + Usz/2, Vmin = Vc - Vsz/2, Vmax = Vc + Vsz/2;

    // Build a continuous list of UVs tracing the rectangle perimeter
    const uv_cont = [];
    for(let i=0;i<=edge;i++){ const t=i/edge; uv_cont.push([Umin+t*(Umax-Umin), Vmax]); }
    for(let i=1;i<=edge;i++){ const t=i/edge; uv_cont.push([Umax, Vmax+t*(Vmin-Vmax)]); }
    for(let i=1;i<=edge;i++){ const t=i/edge; uv_cont.push([Umax+t*(Umin-Umax), Vmin]); }
    for(let i=1;i<edge;i++){ const t=i/edge; uv_cont.push([Umin, Vmin+t*(Vmax-Vmin)]); }

    // Map to plane coordinates for visualization: plane uses [-PI, PI] range along each axis
    const planePts = uv_cont.map(([U,V])=>[(this._wrap01(U)-0.5)*this.TWO_PI, 0, (this._wrap01(V)-0.5)*this.TWO_PI]);

    // Map to torus coordinates using posFromUV
    const torusPts  = uv_cont.map(([U,V])=>{ const P = this.posFromUV(U,V); return [P.x,P.y,P.z]; });

    // Now split the continuous sequence into loops where the plane jumps across seams
    // If a consecutive segment jumps more than PI in either plane axis, we break the loop
    const planeLoops = [], torusLoops = [];
    let curPlane = [], curTorus = [];

    function pushCurrent(){
      if(curPlane.length>0){
        planeLoops.push(curPlane.slice());
        torusLoops.push(curTorus.slice());
      }
      curPlane = []; curTorus = [];
    }

    if(planePts.length>0){ curPlane.push(planePts[0]); curTorus.push(torusPts[0]); }

    for(let i=1;i<planePts.length;i++){
      const a = planePts[i-1], b = planePts[i];
      // difference along plane axes
      const dx = Math.abs(b[0]-a[0]), dz = Math.abs(b[2]-a[2]);
      // If jump is large - typically due to wrap, break the current loop
      if(dx > Math.PI || dz > Math.PI){ pushCurrent(); }
      curPlane.push(b); curTorus.push(torusPts[i]);
    }
    if(curPlane.length>0) pushCurrent();

    return { uv: uv_cont, planePts, torusPts, planeLoops, torusLoops };
  }

  /**
   * Produce a spiral arc along V vs U0
   * - thetaRad given in radians specifying base U location
   * - optional 'close1' mode will close after 1 lap when shear is present
   *
   * @param {number} thetaRad - starting theta in radians
   * @param {number} offset - offset outwards along normalized position (optional)
   * @param {Object} opts - options: {mode:'blue'|'close1', samples:number}
   * @returns {{mode:string, U0:number, k:number, torusPts:Array, planePts:Array, uv:Array}}
   */
  spiralArcData(thetaRad, offset=0, opts={mode:'blue', samples:320}){
    const mode = opts.mode || 'blue';
    const S = opts.samples || 320;
    const U0 = this._wrap01(thetaRad / this.TWO_PI);
    const torusPts = [], planePts = [], uv = [];
    const shear = this.params.shear || 0.0;

    // k is non-zero when we want the U to slide with the arc so that it closes after one lap
    const k = (mode === 'close1') ? (Math.round(shear) - shear) : 0.0;

    for(let i=0;i<=S;i++){
      const t = i / S;
      const U = this._wrap01(U0 + k * t); // apply closure shift if requested
      const V = this._wrap01(t);
      const P = this.posFromUV(U, V);
      const out = P.clone().normalize().multiplyScalar(offset);
      torusPts.push([P.x + out.x, P.y + out.y, P.z + out.z]);
      planePts.push([(this._wrap01(U)-0.5)*this.TWO_PI, 0, (this._wrap01(V)-0.5)*this.TWO_PI]);
      uv.push([U, V]);
    }

    // If requested to close after one lap, duplicate the first point at the end
    if(mode === 'close1'){
      torusPts[torusPts.length-1] = [...torusPts[0]];
      planePts[planePts.length-1] = [...planePts[0]];
      uv[uv.length-1] = [...uv[0]];
    }

    return { mode, U0, k, torusPts, planePts, uv };
  }

  /**
   * Cross-section square around the tube at U = thetaRad
   * - fits a circle in the local tube plane and builds an axis-aligned square around that circle
   * - returns exactly 4 corner points (no duplicate closing point)
   *
   * @param {number} thetaRad - U position in radians
   * @param {number} sizeMultiplier - multiplier of fitted radius used for half-size
   * @returns {{
   *   U:number,
   *   center:Array<number>,
   *   axes:{x:Array,y:Array,n:Array},
   *   torusPts:Array<Array<number>>,
   *   planePts:Array<Array<number>>
   * }}
   */
  crossSquareData(thetaRad, sizeMultiplier=1.0){
    const U = this._wrap01(thetaRad / this.TWO_PI);
    const S = 256, eps = 1e-3;

    // Sample tube ring points P(U, v) for v in [0,1)
    const Ps = new Array(S);
    let C0 = new THREE.Vector3(0,0,0);
    for(let i=0;i<S;i++){ const V=i/S; const P=this.posFromUV(U,V); Ps[i]=P; C0.add(P); }
    C0.multiplyScalar(1/S); // centroid of sampled ring - fallback center

    // Estimate plane normal by averaging cross products of successive centered vectors
    let n = new THREE.Vector3(0,0,0);
    for(let i=0;i<S;i++){
      const a = Ps[i].clone().sub(C0);
      const b = Ps[(i+1)%S].clone().sub(C0);
      n.add(a.clone().cross(b));
    }
    // If degeneracy occurs, fallback to derivative-based normal
    if(n.lengthSq() < 1e-12){
      const P0 = this.posFromUV(U,0.0);
      const Pv = this.posFromUV(U, eps);
      n = Pv.clone().sub(P0).cross(P0.clone().sub(this.posFromUV(U+eps,0))).normalize();
      if(n.lengthSq() < 1e-12) n = new THREE.Vector3(0,1,0);
    }
    n.normalize();

    // Choose local axes ex and ey spanning the plane
    // ex is computed by averaging radial-type vectors projected to the plane
    let ex = new THREE.Vector3(0,0,0);
    for(let i=0;i<S;i++){
      const v = Ps[i].clone().sub(C0);
      const proj = n.clone().multiplyScalar(v.dot(n));
      ex.add(v.clone().sub(proj));
    }
    if(ex.lengthSq() < 1e-12){
      if(Math.abs(n.x) < 0.9) ex = new THREE.Vector3(1,0,0).cross(n).normalize();
      else ex = new THREE.Vector3(0,1,0).cross(n).normalize();
    } else ex.normalize();

    const ey = n.clone().cross(ex).normalize();

    // Project sampled points into plane coordinates to fit a circle using Kasa method
    const xs = new Float64Array(S), ys = new Float64Array(S);
    for(let i=0;i<S;i++){
      const v = Ps[i].clone().sub(C0);
      xs[i] = v.dot(ex);
      ys[i] = v.dot(ey);
    }

    // Build normal equations to solve xi^2 + yi^2 + A xi + B yi + C = 0
    let M = [[0,0,0],[0,0,0],[0,0,0]];
    let rhs = [0,0,0];
    for(let i=0;i<S;i++){
      const xi = xs[i], yi = ys[i];
      const q = -(xi*xi + yi*yi);
      M[0][0] += xi*xi; M[0][1] += xi*yi; M[0][2] += xi;
      M[1][0] += xi*yi; M[1][1] += yi*yi; M[1][2] += yi;
      M[2][0] += xi;    M[2][1] += yi;    M[2][2] += 1;
      rhs[0] += xi*q; rhs[1] += yi*q; rhs[2] += q;
    }

    let sol = solve3x3(M, rhs);
    let center2 = { x:0, y:0 }, radius = 0;
    if(sol){
      const A = sol[0], B = sol[1], Cc = sol[2];
      center2.x = -A/2; center2.y = -B/2;
      const rad2 = center2.x*center2.x + center2.y*center2.y - Cc;
      radius = (rad2 > 0) ? Math.sqrt(rad2) : Math.sqrt(Math.max(0, (xs.reduce((s,v)=>s+v*v,0)+ys.reduce((s,v)=>s+v*v,0))/S));
    } else {
      // fallback to centroid radius
      let meanR = 0;
      for(let i=0;i<S;i++) meanR += Math.hypot(xs[i], ys[i]);
      meanR /= S;
      center2.x = 0; center2.y = 0;
      radius = meanR;
    }

    // Map fit center back to 3D
    const C = C0.clone().add(ex.clone().multiplyScalar(center2.x)).add(ey.clone().multiplyScalar(center2.y));

    // Build orthonormal square axes
    let xAxis = new THREE.Vector3(0,0,0);
    for(let i=0;i<S;i++){ const v = Ps[i].clone().sub(C); xAxis.add(v); }
    if(xAxis.lengthSq() < 1e-12) xAxis = ex.clone(); else xAxis.normalize();

    // Project xAxis to plane to ensure orthogonality to normal n
    const proj = n.clone().multiplyScalar(xAxis.dot(n));
    xAxis.sub(proj);
    if(xAxis.lengthSq() < 1e-12){
      if(Math.abs(n.x) < 0.9) xAxis = new THREE.Vector3(1,0,0).cross(n).normalize();
      else xAxis = new THREE.Vector3(0,1,0).cross(n).normalize();
    } else xAxis.normalize();

    const yAxis = n.clone().cross(xAxis).normalize();

    // Half-size scaled by fitted radius
    const h = Math.max(1e-9, sizeMultiplier) * radius;

    // Compute four corners (no duplicate closing point)
    const A3 = C.clone().add(xAxis.clone().multiplyScalar( h)).add(yAxis.clone().multiplyScalar( h));
    const B3 = C.clone().add(xAxis.clone().multiplyScalar( h)).add(yAxis.clone().multiplyScalar(-h));
    const D3 = C.clone().add(xAxis.clone().multiplyScalar(-h)).add(yAxis.clone().multiplyScalar( h));
    const E3 = C.clone().add(xAxis.clone().multiplyScalar(-h)).add(yAxis.clone().multiplyScalar(-h));

    const torusPts = [A3, B3, E3, D3].map(v=>[v.x,v.y,v.z]); // exactly 4 corners

    // planePts for cross visual: sample vertical line in plane for reference
    const planePts = [];
    for(let i=0;i<=S;i++){ const V=i/S; planePts.push([(U-0.5)*this.TWO_PI,0,(V-0.5)*this.TWO_PI]); }
    return { U, center:[C.x,C.y,C.z], axes:{x:[xAxis.x,xAxis.y,xAxis.z], y:[yAxis.x,yAxis.y,yAxis.z], n:[n.x,n.y,n.z]}, torusPts, planePts };
  }

  /**
   * uvFromXYZ_std
   * - inverse map for standard torus: given 3D position in torus local space,
   *   compute fractional (u,v) coordinates on the parametric domain.
   * - returns THREE.Vector2(Ufraction, Vfraction)
   */
  uvFromXYZ_std(pos){
    if(this.params.mode !== 'std') return null;
    const x = pos.x, y = pos.z, z = pos.y; // note swapped ordering used across code
    const u = Math.atan2(y, x);
    const Rxy = Math.hypot(x, y);
    const cv = (Rxy - this.params.R) / this.params.r;
    const v = Math.atan2(z / this.params.r, cv);
    return new THREE.Vector2(this._wrap01(u / this.TWO_PI), this._wrap01(v / this.TWO_PI));
  }
}
