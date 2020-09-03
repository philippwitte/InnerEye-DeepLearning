
# A map between PL system names and 2-tuples with elements:
#   elem 1: PL system module name
#   elem 2: name of argparser function to apply to argments passed to module
PL_MODULES = {
    "cpc_stl10_train": ("InnerEye.Research.cpc.pl_systems.cpc_stl10_train",
                        "cpc_stl10_argparser"),
    "cpc_med_train": ("InnerEye.Research.cpc.pl_systems.cpc_med_train",
                      "cpc_med_argparser"),
    "cpc_stl10_classifier": ("InnerEye.Research.cpc.pl_systems.cpc_stl10_classifier",
                              "cpc_downstream_clf_argparser"),
    "cpc_med_encode": ("InnerEye.Research.cpc.pl_systems.cpc_med_encode",
                       "cpc_med_encode_argparser"),
    "cpc_dual_view_med_train": ("InnerEye.Research.cpc.pl_systems.cpc_dual_view_med_train",
                                "cpc_dual_view_med_argparser"),
    "cpc_tri_view_med_train": ("InnerEye.Research.cpc.pl_systems.cpc_tri_view_med_train",
                               "cpc_tri_view_med_argparser"),
    "med_dual_view_supervised": ("InnerEye.Research.cpc.pl_systems.med_dual_view_supervised",
                                 "med_dual_view_supervised")
}
