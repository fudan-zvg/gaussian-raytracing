import trimesh
from gaussian import GaussianModel

string = """
out idx: 458776, gs_idx:191084
out idx: 458776, gs_idx:225157
out idx: 458776, gs_idx:153964
out idx: 458776, gs_idx:226801
out idx: 458776, gs_idx:157504
out idx: 458776, gs_idx:116538
out idx: 458776, gs_idx:244761
out idx: 458776, gs_idx:244761
out idx: 458776, gs_idx:206960
out idx: 458776, gs_idx:5973
out idx: 458776, gs_idx:157605
out idx: 458776, gs_idx:93657
out idx: 458776, gs_idx:100954
out idx: 458776, gs_idx:192180
out idx: 458776, gs_idx:242576
out idx: 458776, gs_idx:251295
out idx: 458776, gs_idx:126403
out idx: 458776, gs_idx:214032
out idx: 458776, gs_idx:203199
out idx: 458776, gs_idx:196861
out idx: 458776, gs_idx:202833
out idx: 458776, gs_idx:165191
out idx: 458776, gs_idx:44626
out idx: 458776, gs_idx:44626
out idx: 458776, gs_idx:110148
out idx: 458776, gs_idx:244021
out idx: 458776, gs_idx:195331
out idx: 458776, gs_idx:152573
out idx: 458776, gs_idx:301831
out idx: 458776, gs_idx:217169
out idx: 458776, gs_idx:200377
out idx: 458776, gs_idx:200377
out idx: 458776, gs_idx:159784
out idx: 458776, gs_idx:259317
out idx: 458776, gs_idx:178759
out idx: 458776, gs_idx:98307
out idx: 458776, gs_idx:175687
out idx: 458776, gs_idx:173998
out idx: 458776, gs_idx:75356
out idx: 458776, gs_idx:132841
out idx: 458776, gs_idx:111365
out idx: 458776, gs_idx:179141
out idx: 458776, gs_idx:102293
out idx: 458776, gs_idx:96959
out idx: 458776, gs_idx:197986
out idx: 458776, gs_idx:209494
out idx: 458776, gs_idx:75831
out idx: 458776, gs_idx:180917
out idx: 458776, gs_idx:175397
out idx: 458776, gs_idx:168618
out idx: 458776, gs_idx:235222
out idx: 458776, gs_idx:188994
out idx: 458776, gs_idx:253242
out idx: 458776, gs_idx:142667
out idx: 458776, gs_idx:156131
out idx: 458776, gs_idx:156131
out idx: 458776, gs_idx:79442
out idx: 458776, gs_idx:132669
out idx: 458776, gs_idx:150249
out idx: 458776, gs_idx:85118
out idx: 458776, gs_idx:221517
out idx: 458776, gs_idx:156545
out idx: 458776, gs_idx:44440
out idx: 458776, gs_idx:44440
out idx: 458776, gs_idx:212656
out idx: 458776, gs_idx:8219
out idx: 458776, gs_idx:157820
out idx: 458776, gs_idx:295286
out idx: 458776, gs_idx:55373
out idx: 458776, gs_idx:164806
out idx: 458776, gs_idx:221565
out idx: 458776, gs_idx:221565
out idx: 458776, gs_idx:239502
out idx: 458776, gs_idx:161370
out idx: 458776, gs_idx:54817
out idx: 458776, gs_idx:217356
out idx: 458776, gs_idx:221378
out idx: 458776, gs_idx:237898
out idx: 458776, gs_idx:234491
out idx: 458776, gs_idx:152223
out idx: 458776, gs_idx:138736
out idx: 458776, gs_idx:243915
out idx: 458776, gs_idx:214379
out idx: 458776, gs_idx:97727
out idx: 458776, gs_idx:105869
out idx: 458776, gs_idx:205478
out idx: 458776, gs_idx:49090
out idx: 458776, gs_idx:49090
out idx: 458776, gs_idx:208710
out idx: 458776, gs_idx:209438
out idx: 458776, gs_idx:175225
out idx: 458776, gs_idx:89100
out idx: 458776, gs_idx:187328
out idx: 458776, gs_idx:138097
out idx: 458776, gs_idx:42
out idx: 458776, gs_idx:42
out idx: 458776, gs_idx:62357
out idx: 458776, gs_idx:12929
out idx: 458776, gs_idx:42747
out idx: 458776, gs_idx:51822
out idx: 458776, gs_idx:25028
out idx: 458776, gs_idx:6855
out idx: 458776, gs_idx:26636
out idx: 458776, gs_idx:21451
out idx: 458776, gs_idx:57184
out idx: 458776, gs_idx:21729
out idx: 458776, gs_idx:36971
out idx: 458776, gs_idx:46358
out idx: 458776, gs_idx:12071
out idx: 458776, gs_idx:63097
out idx: 458776, gs_idx:3667
out idx: 458776, gs_idx:3667
out idx: 458776, gs_idx:12905
out idx: 458776, gs_idx:16572
out idx: 458776, gs_idx:15198
out idx: 458776, gs_idx:109905
out idx: 458776, gs_idx:16204
out idx: 458776, gs_idx:33108
out idx: 458776, gs_idx:4049
out idx: 458776, gs_idx:4049
out idx: 458776, gs_idx:15202
out idx: 458776, gs_idx:69296
out idx: 458776, gs_idx:1685
out idx: 458776, gs_idx:134000
out idx: 458776, gs_idx:44064
out idx: 458776, gs_idx:77080
out idx: 458776, gs_idx:40823
out idx: 458776, gs_idx:4100
out idx: 458776, gs_idx:127500
out idx: 458776, gs_idx:55671
out idx: 458776, gs_idx:12962
out idx: 458776, gs_idx:40091
out idx: 458776, gs_idx:36803
out idx: 458776, gs_idx:77274
out idx: 458776, gs_idx:11703
out idx: 458776, gs_idx:93257
out idx: 458776, gs_idx:4559
out idx: 458776, gs_idx:62330
out idx: 458776, gs_idx:3397
out idx: 458776, gs_idx:12629
out idx: 458776, gs_idx:175089
out idx: 458776, gs_idx:9168
"""

# extract the indices in string, after "i_prim", using regex
import re
indices = list(map(int, re.findall(r"gs_idx:(\d+)", string)))
print(indices)


gaussians = GaussianModel(3)
gaussians.load_ply("point_cloud.ply")
xyz = gaussians.get_xyz

# import pdb;pdb.set_trace()
mesh = trimesh.Trimesh(xyz[indices].detach().cpu().numpy()).export("test2.ply")