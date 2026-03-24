# Krabby-Uno: Blender → USD pipeline

This document records the **implemented** path used to produce the authoritative hexapod USD for Isaac Sim from the Blender source in this directory.

## Purpose and authority

| Role | Path |
|------|------|
| **Blender source** | `assets/Krabby-Uno.blend` (saved with **Blender 5.1**) |
| **Authoritative USD** | `assets/crab_hex.usd` — use this going forward for training and evaluation workflows that expect the full Krabby-Uno asset from Blender |
| **Reference USD (testing)** | `assets/crab_hex_ref.usd` — produced from `assets/crab_hex_ref.urdf`; not authoritative; for tests and experiments only |

## Why this pipeline

USD for Isaac Sim is exported using the **[NVIDIA Omniverse Blender (Alpha USD) build](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/omni_blender)**. That build aligns with Omniverse/Isaac Sim expectations better than exporting USD from stock Blender alone for this workflow.

The Alpha build used here is based on **Blender 4.2**, while the robot is modeled in **Blender 5.x**. A **5.1** `.blend` file is not reliably opened in the 4.2-based Alpha build, so the mesh is moved through **FBX**: author in 5.1, export FBX, import in the Alpha build, then export **Universal Scene Description** from there. Articulation, colliders, masses, and joint limits are completed in **Isaac Sim** on the resulting USD.

## Blender → USD procedure

1. Open **`assets/Krabby-Uno.blend`** in **Blender 5.1** (matching the save version).
2. **File → Export → FBX** and save an `.fbx` file. Use project-appropriate FBX options (scale, selected objects vs scene, mesh/armature). If you use defaults, document any issues (scale, missing meshes) when importing in step 3.
3. Open **NVIDIA Blender Alpha (Omniverse), Blender 4.2** build.
4. **File → Import** the FBX from step 2.
5. **File → Export → Universal Scene Description** and save a `.usd` (or `.usda` / `.usdc` as needed). This USD is the starting point for Isaac Sim rigging below.

## Isaac Sim: post-import rigging

After loading the exported USD in **Isaac Sim**, the following was applied so the robot simulates as an articulation with the intended topology:

1. Set **`/World/KrabbyUno`** as the **articulation root**.
2. Add **Collider** and **Rigid Body** for the relevant **leg** parts (as needed for stable contact and dynamics).
3. **Fixed joint:** attach **top plate** and **bottom plate**.
4. **Fixed joint:** attach each **HipMount** to the **bottom plate**.
5. **Revolute joint:** **Hip mount** ↔ **Hip**.
6. **Prismatic joint:** **Hip** ↔ **Femur**.
7. **Prismatic joint:** **Femur** ↔ **Tibia**.
8. Assign **mass** to all leg parts and to the **top** and **bottom** plates.
9. Set **joint limits** and **prismatic constraints** as required for stable motion and hardware-consistent ranges.


## References

- [NVIDIA NGC — Omniverse Blender (Alpha USD)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/omni_blender)
- [Omniverse Blender Connector — Manual](https://docs.omniverse.nvidia.com/connect/latest/blender/manual.html)
