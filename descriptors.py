import numpy as np
from pymatgen.core import Element, Structure


def extract_sendek_descriptors(cif_content: str) -> dict:
    """Parses a CIF string and extracts the 5 Sendek crystal descriptors.

    Descriptors:
      - LLSD: Shortest Li-Li distance (Å)
      - LASD: Shortest Li-anion distance (Å)
      - LLB : Local Li bottleneck metric
      - AFC : Average framework coordination
      - SBI : Sublattice packing / bonding index
    """
    try:
        struct = Structure.from_str(cif_content, fmt="cif")
        # Merge partially occupied or split sites within 0.15 Å
        struct.merge_sites(tol=0.15, mode="delete")
    except Exception as e:
        raise ValueError(f"Failed to parse CIF structure: {e}")

    li_indices = [i for i, site in enumerate(struct) if site.specie == Element("Li")]
    if not li_indices:
        raise ValueError("Invalid material: Crystal structure contains no Lithium (Li) sites.")

    anion_species = {
        Element("O"), Element("S"), Element("Se"), Element("Te"),
        Element("N"), Element("P"), Element("F"), Element("Cl"),
        Element("Br"), Element("I")
    }

    # --- 1. LLSD (Li-Li Shortest Distance) ---
    all_li_distances = []
    for i in li_indices:
        site = struct[i]
        # n.nn_distance accounts for periodic boundary conditions
        neighbors = struct.get_neighbors(site, r=8.0)
        li_dists = [
            float(n.nn_distance)
            for n in neighbors
            if n.specie == Element("Li") and n.nn_distance > 0.15  # filter self & duplicate sites
        ]
        if li_dists:
            all_li_distances.append(min(li_dists))

    # Fallback to training median if material has isolated single Li
    llsd = float(min(all_li_distances)) if all_li_distances else 2.9760

    # --- 2. LASD (Li-Anion Shortest Distance) ---
    all_la_distances = []
    for i in li_indices:
        site = struct[i]
        neighbors = struct.get_neighbors(site, r=6.0)
        la_dists = [
            float(n.nn_distance)
            for n in neighbors
            if n.specie in anion_species and n.nn_distance > 0.15
        ]
        if la_dists:
            all_la_distances.append(min(la_dists))

    lasd = float(min(all_la_distances)) if all_la_distances else 2.3977

    # --- 3. LLB (Li-Li shortest distance / Li-neighbor distance metric) ---
    # Captures path bottleneck constraints between local sites
    llb = float(np.clip(llsd / max(lasd, 1.0), 1.0, 8.0))

    # --- 4. AFC (Average Framework Coordination) ---
    framework_indices = [i for i in range(len(struct)) if i not in li_indices]
    if framework_indices:
        coord_counts = []
        for i in framework_indices:
            neighbors = struct.get_neighbors(struct[i], r=3.5)
            # Count distinct non-self neighbors
            valid_neighbors = [n for n in neighbors if n.nn_distance > 0.15]
            coord_counts.append(len(valid_neighbors))
        afc = float(np.mean(coord_counts) / max(len(struct), 1.0))
    else:
        afc = 0.0206

    # --- 5. SBI (Sublattice Bonding / Packing Index) ---
    v_cell = struct.volume
    v_occupied = sum(
        (site.specie.atomic_radius or 1.0)**3 * (4.0 / 3.0) * np.pi
        for site in struct
    )
    packing_fraction = v_occupied / max(v_cell, 1.0)
    sbi = float(packing_fraction * 25.0)

    return {
        "SBI": round(sbi, 4),
        "AFC": round(afc, 4),
        "LLSD": round(llsd, 4),
        "LLB": round(llb, 4),
        "LASD": round(lasd, 4),
    }