/**
 * Quaternion (qx, qy, qz, qw) from intrinsic Euler angles in **zyx** order
 * (same convention as `scipy.spatial.transform.Rotation.from_euler('zyx', [α, β, γ])`,
 * result layout `as_quat()` → [x, y, z, w]).
 *
 * Here α, β, γ map to Expo `DeviceMotion.rotation` alpha, beta, gamma (radians).
 */
export function eulerZyxRadiansToQuatXyzw(
  alpha: number,
  beta: number,
  gamma: number,
): { qx: number; qy: number; qz: number; qw: number } {
  const ha = alpha * 0.5;
  const hb = beta * 0.5;
  const hc = gamma * 0.5;
  const ca = Math.cos(ha);
  const sa = Math.sin(ha);
  const cb = Math.cos(hb);
  const sb = Math.sin(hb);
  const cc = Math.cos(hc);
  const sc = Math.sin(hc);
  return {
    qx: ca * cb * sc + sa * sb * cc,
    qy: ca * sb * cc - sa * cb * sc,
    qz: sa * cb * cc + ca * sb * sc,
    qw: ca * cb * cc - sa * sb * sc,
  };
}
