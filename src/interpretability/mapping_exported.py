feature_to_cluster = {
    # Early transition-metal oxides/fluorides
    3479: "Early transition-metal oxides/fluorides",
    482:  "Early transition-metal oxides/fluorides",
    2708: "Early transition-metal oxides/fluorides",
    2201: "Early transition-metal oxides/fluorides",
    3429: "Early transition-metal oxides/fluorides",
    1862: "Early transition-metal oxides/fluorides",
    3850: "Early transition-metal oxides/fluorides",
    684:  "Early transition-metal oxides/fluorides",

    # Polyanionic/complex oxide networks (Cluster B)
    2871: "Polyanionic networks",
    2454: "Polyanionic networks",
    3440: "Polyanionic networks",
    872:  "Polyanionic networks",
    740:  "Polyanionic networks",
    1015: "Polyanionic networks",
    3128: "Polyanionic networks",

    # Actinide compounds (Cluster C)
    3430: "Actinide compounds",
    2425: "Actinide compounds",
    519:  "Actinide compounds",
    1488: "Actinide compounds",
    467:  "Actinide compounds",
    95:   "Actinide compounds",
    2615: "Actinide compounds",

    # Refractory/noble transition metals + carbides/borides (Cluster D)
    418:  "Refractory transition metals",
    2105: "Carbides/silicides/borides",
    3374: "Noble/platinum-group metals",
    2470: "Intermetallics and post-transition metals",
    3089: "Refractory transition metals",
    871:  "Intermetallics and post-transition metals",
    1063: "Noble/platinum-group metals",
    510:  "Refractory transition metals",
    3318: "Noble/platinum-group metals",
    1449: "Refractory transition metals",
    943:  "Refractory transition metals",
    3077: "Intermetallics and post-transition metals",
    2800: "Noble/platinum-group metals",
    983:  "Carbides/silicides/borides",
    2174: "Refractory transition metals",
    371:  "Refractory transition metals",

    # Alkali/alkaline-earth halides (Cluster E)
    3500: "Alkali/alkaline-earth halides",
    575:  "Alkali/alkaline-earth halides",
    4066: "Alkali/alkaline-earth halides",
    1048: "Alkali/alkaline-earth halides",
    742:  "Alkali/alkaline-earth halides",
    2348: "Alkali/alkaline-earth halides",
    192:  "Alkali/alkaline-earth halides",
    3320: "Alkali/alkaline-earth halides",

    # Rare-earth/lanthanide compounds (Cluster F)
    2421: "Rare-earth/lanthanide compounds",
    3519: "Rare-earth/lanthanide compounds",
    3887: "Rare-earth/lanthanide compounds",
    3790: "Rare-earth/lanthanide compounds",
    1811: "Rare-earth/lanthanide compounds",
    2611: "Rare-earth/lanthanide compounds",
    1149: "Rare-earth/lanthanide compounds",
    1930: "Rare-earth/lanthanide compounds",
}

feature_secondary_clusters: dict[int, list[str]] = {
    # Features spanning actinide + structural complexity
    3430: ["Structurally complex/low-symmetry"],
    2425: ["Heavy complex oxides"],
    1488: ["Heavy complex oxides"],

    # Polyanionic features with structural complexity
    2871: ["Structurally complex/low-symmetry"],
    2454: ["Structurally complex/low-symmetry"],

    # Carbide/boride features also appearing in refractory metals context
    2105: ["Refractory transition metals"],
    983:  ["Refractory transition metals"],
}

# label_to_cluster = {
#     feature_labels[fid]: feature_to_cluster.get(fid, "Unassigned")
#     for fid in feature_labels
# }

# join_table = [
#     {
#         "feature_id": fid,
#         "label":      feature_labels[fid],
#         "cluster":    feature_to_cluster.get(fid, "Unassigned"),
#     }
#     for fid in sorted(feature_labels)
# ]

# if __name__ == "__main__":
#     import json, csv, io

#     print("=== label_to_cluster ===")
#     print(json.dumps(label_to_cluster, indent=2))

#     print("\n=== join_table (feature_id | label | cluster) ===")
#     print(json.dumps(join_table, indent=2))

#     buf = io.StringIO()
#     writer = csv.DictWriter(buf, fieldnames=["feature_id", "label", "cluster"])
#     writer.writeheader()
#     writer.writerows(join_table)
#     print("\n--- CSV preview (first 10 rows) ---")
#     print("\n".join(buf.getvalue().splitlines()[:11]))