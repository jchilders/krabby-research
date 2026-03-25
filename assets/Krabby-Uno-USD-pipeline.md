# Krabby-Uno: Blender → USD pipeline

This document records the **implemented** path used to produce the authoritative hexapod USD for Isaac Sim from the Blender source in this directory.

## Purpose and authority

| Role | Path |
|------|------|
| **Blender source** | `assets/Krabby-Uno.blend` (saved with **Blender 5.1**) |
| **Canonical robot USD (checked in)** | **`assets/crab_hex.usd`** — the asset to load for sim, scripts, and anything that references the Krabby-Uno hexapod stage (`/World/KrabbyUno`). |
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

1. Set **`/World/KrabbyUno`** as the **articulation root** (with `xformOp:translate` / `xformOp:orient` / `xformOp:scale` as in the file).
2. Add **Collider** and **Rigid Body** for the relevant **leg** parts (as needed for stable contact and dynamics).
3. **Fixed joint:** attach **top plate** and **bottom plate** (including **`Plate_Weld_FixedJoint`**-style joints with aligned local frames — identity offset/rotation where needed so the stack does not pitch incorrectly).
4. **Fixed joint:** attach each **HipMount** to the **bottom plate**.
5. **Revolute joint:** **HipMount** ↔ **Hip** (yaw).
6. **Hip–Femur chain (per leg):** same “slider body + prismatic + revolute” pattern as the knee. The implemented graph uses:
   - **`FemurPrismatic`** rigid under each leg root (e.g. `Root_MR/MR_FemurPrismatic`).
   - **Prismatic joint:** **Hip** ↔ **`FemurPrismatic`** (extension along the joint axis; **`Hip_FemurPrismatic_PrismaticJoint`** in the file).
   - **Revolute joint:** **`FemurPrismatic`** ↔ **Femur** (with **`physics:excludeFromArticulation`** on the helper rigid where required). In-repo: **tight angular limits ±0.001°**, no angular drive (non-actuated DOF).
   - **Revolute joint:** **Hip** ↔ **Femur** (**`Hip_Femur_RevoluteJoint`**). In-repo: **tight angular limits ±0.001°**, no angular drive.
7. **Knee chain (per leg):** not a single Femur–Tibia prismatic alone. The implemented graph uses:
   - **`TibiaPrismatic`** rigid under each leg root (e.g. `Root_MR/MR_TibiaPrismatic`) as a **slider body** between Femur and Tibia.
   - **Prismatic joint:** **Femur** ↔ **`TibiaPrismatic`** (knee extension along the joint axis, with linear drives and limits).
   - **Revolute joint:** **`TibiaPrismatic`** ↔ **Tibia** (with **`physics:excludeFromArticulation`** on the helper rigid bodies where required so the articulation stays well-formed). In-repo: **tight angular limits ±0.001°**, no angular drive.
   - **Revolute joint:** **Femur** ↔ **Tibia** (**`Femur_Tibia_RevoluteJoint`**) with **tight angular limits ±0.001°** and **no angular drive**; knee extension is commanded on the **Femur–Tibia prismatic** linear drive only (**FR / FL / RR / RL / MR / ML**).
8. Assign **mass** to all leg parts and to the **top** and **bottom** plates (avoid **duplicate** mass on decorative or child meshes that share the same physical body).
9. Set **joint limits** and **drive** parameters (stiffness, damping, targets) for stable motion and hardware-consistent ranges — hip **angular** and hip/knee **prismatic** drives use **moderate stiffness with higher damping** in-repo to avoid idle oscillation under gravity/contacts; knee **linear** drives use **target position 0** where appropriate.

**Actuation model (3 motors per leg, in-repo):** only **`HipRevoluteJoint`** has an **angular** drive; **hip–femur** and **femur–tibia** **prismatic** joints have **linear** drives. All other leg revolutes are **passive** with **±0.001°** angular limits (no `PhysicsDriveAPI:angular`). See `assets/scripts/README.md` for the **Flat18** command layout. These details apply to the checked-in **`assets/crab_hex.usd`**.

### In-repo USD edits (after Isaac Sim)

The checked-in **`crab_hex.usd`** includes **hand-tuned physics** not implied by “export once from Blender”: joint drive gains, plate-weld frame cleanup, **passive** (limit-only) revolutes on the parallel leg chain, mirrored **MR/ML** leg graphs to match the other four legs, and mass fixes. When regenerating from Blender/Isaac, re-apply or merge these layers rather than expecting a fresh export to match bit-for-bit.


## References

- [NVIDIA NGC — Omniverse Blender (Alpha USD)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/omni_blender)
- [Omniverse Blender Connector — Manual](https://docs.omniverse.nvidia.com/connect/latest/blender/manual.html)
