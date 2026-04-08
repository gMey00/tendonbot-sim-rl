import omni.usd
from pxr import Usd, UsdPhysics, Sdf

stage = omni.usd.get_context().get_stage()

CLOTH_MESH = Sdf.Path("/World/tshirt_mod/World/mesh/Mesh")   # <-- mesh prim
PHYS_MAT   = Sdf.Path("/World/PhysicsMaterials/cloth_mat")

# Create container + physics material prim
UsdPhysics.MaterialAPI.Apply(stage.DefinePrim("/World/PhysicsMaterials", "Xform"))

mat_prim = stage.DefinePrim(str(PHYS_MAT), "Material")
mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)

# Set values (realistic cloth-ish)
mat_api.CreateStaticFrictionAttr().Set(0.4)
mat_api.CreateDynamicFrictionAttr().Set(0.3)
mat_api.CreateRestitutionAttr().Set(0.0)

# Bind to cloth mesh
mesh_prim = stage.GetPrimAtPath(CLOTH_MESH)
UsdPhysics.MaterialBindingAPI.Apply(mesh_prim).Bind(mat_prim)

print("Bound physics material", PHYS_MAT, "to", CLOTH_MESH)