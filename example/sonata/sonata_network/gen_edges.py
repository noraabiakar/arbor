import h5py

f = h5py.File("edges_brunel.h5", "a")

edges = f.create_group("edges")

pop_e_e = edges.create_group("pop_e_e")
pop_i_e = edges.create_group("pop_i_e")
pop_i_i = edges.create_group("pop_i_i")
pop_e_i = edges.create_group("pop_e_i")

#################################################################################
nedges = 8000
ntgts = 400
nsrcs = 400
s2t_ratio = nsrcs//ntgts #1
nedges_per_tgt = nedges//ntgts #20
nedges_per_src = nedges//nsrcs #20 

edge_group_id = pop_e_e.create_dataset("edge_group_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_id[i] = 0

edge_group_index = pop_e_e.create_dataset("edge_group_index", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_index[i] = i

edge_type_id = pop_e_e.create_dataset("edge_type_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_type_id[i] = 100

source_node_id = pop_e_e.create_dataset("source_node_id", (nedges,), dtype="i")
for i in range(0,nsrcs):
    for j in range(0,nedges_per_src):
        source_node_id[i*nedges_per_src + j] = (i + j + 1) % nsrcs

target_node_id = pop_e_e.create_dataset("target_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range(0,nedges_per_tgt):
        target_node_id[i*nedges_per_tgt + j] = i

ind = pop_e_e.create_group("indicies")
source_to_target = ind.create_group("source_to_target")

node_id_to_ranges = source_to_target.create_dataset("node_id_to_ranges", (nsrcs,2), dtype="i")
for i in range(0,nsrcs):
    node_id_to_ranges[i] = i*nedges_per_src, (i+1)*nedges_per_src

range_to_edge_id = source_to_target.create_dataset("range_to_edge_id", (nedges,2), dtype="i")
for i in range(0,nsrcs):
    for j in range(0, nedges_per_src):
        t = (i - j + nsrcs - 1)%nsrcs
        range_to_edge_id[i*nedges_per_src + j] = t*nedges_per_src + j, t*nedges_per_src + j + 1
         
target_to_source = ind.create_group("target_to_source")

node_id_to_ranges = target_to_source.create_dataset("node_id_to_ranges", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    node_id_to_ranges[i] = i, i+1

range_to_edge_id = target_to_source.create_dataset("range_to_edge_id", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    range_to_edge_id[i] = i*nedges_per_tgt, (i+1)*nedges_per_tgt



g0 = pop_e_e.create_group("0")

afferent_id = g0.create_dataset("afferent_section_id", (nedges,), dtype="i")
afferent_pos = g0.create_dataset("afferent_section_pos", (nedges,), dtype="f")
efferent_id = g0.create_dataset("efferent_section_id", (nedges,), dtype="i")
efferent_pos = g0.create_dataset("efferent_section_pos", (nedges,), dtype="f")

for i in range(0,nedges):
    afferent_id[i] = 0
    afferent_pos[i] = 0.5
    efferent_id[i] = 0
    efferent_pos[i] = 0.9

##################################################################################
nedges = 2000
ntgts = 400
nsrcs = 100
t2s_ratio = ntgts//nsrcs #4
nedges_per_tgt = nedges//ntgts #5
nedges_per_src = nedges//nsrcs #20 

edge_group_id = pop_i_e.create_dataset("edge_group_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_id[i] = 0

edge_group_index = pop_i_e.create_dataset("edge_group_index", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_index[i] = i

edge_type_id = pop_i_e.create_dataset("edge_type_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_type_id[i] = 101

source_node_id = pop_i_e.create_dataset("source_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range(0,nedges_per_tgt):
        source_node_id[i*nedges_per_tgt + j] = (i + j) % nsrcs 

target_node_id = pop_i_e.create_dataset("target_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range(0,nedges_per_tgt):
        target_node_id[i*nedges_per_tgt + j] = i


ind = pop_i_e.create_group("indicies")
source_to_target = ind.create_group("source_to_target")

node_id_to_ranges = source_to_target.create_dataset("node_id_to_ranges", (nsrcs,2), dtype="i")
for i in range(0,nsrcs):
    node_id_to_ranges[i] = i*nedges_per_src, (i+1)*nedges_per_src

range_to_edge_id = source_to_target.create_dataset("range_to_edge_id", (nedges,2), dtype="i")
#needs to be split in 3
for i in range(0,nsrcs):
    for j in range(0, t2s_ratio):
        for k in range(0, nedges_per_tgt):
            t = (i- k)%nsrcs + (j*nsrcs)
            range_to_edge_id[i*t2s_ratio*nedges_per_tgt + j*nedges_per_tgt + k] = t*nedges_per_tgt + k, t*nedges_per_tgt + k + 1
         
target_to_source = ind.create_group("target_to_source")

node_id_to_ranges = target_to_source.create_dataset("node_id_to_ranges", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    node_id_to_ranges[i] = i, i+1

range_to_edge_id = target_to_source.create_dataset("range_to_edge_id", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    range_to_edge_id[i] = i*nedges_per_tgt, (i+1)*nedges_per_tgt


g0 = pop_i_e.create_group("0")

afferent_id = g0.create_dataset("afferent_section_id", (nedges,), dtype="i")
afferent_pos = g0.create_dataset("afferent_section_pos", (nedges,), dtype="f")
efferent_id = g0.create_dataset("efferent_section_id", (nedges,), dtype="i")
efferent_pos = g0.create_dataset("efferent_section_pos", (nedges,), dtype="f")

for i in range(0,nedges):
    afferent_id[i] = 0
    afferent_pos[i] = 0.5
    efferent_id[i] = 0
    efferent_pos[i] = 0.9

##################################################################################
nedges = 500
ntgts = 100
nsrcs = 100
s2t_ratio = nsrcs//ntgts #1
nedges_per_tgt = nedges//ntgts #5
nedges_per_src = nedges//nsrcs #5


edge_group_id = pop_i_i.create_dataset("edge_group_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_id[i] = 0

edge_group_index = pop_i_i.create_dataset("edge_group_index", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_index[i] = i

edge_type_id = pop_i_i.create_dataset("edge_type_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_type_id[i] = 102

source_node_id = pop_i_i.create_dataset("source_node_id", (nedges,), dtype="i")
for i in range(0,nsrcs):
    for j in range(0,nedges_per_src):
        source_node_id[i*nedges_per_src + j] = (i + j + 1) % nsrcs 

target_node_id = pop_i_i.create_dataset("target_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range(0,nedges_per_tgt):
        target_node_id[i*nedges_per_tgt + j] = i


ind = pop_i_i.create_group("indicies")
source_to_target = ind.create_group("source_to_target")

node_id_to_ranges = source_to_target.create_dataset("node_id_to_ranges", (nsrcs,2), dtype="i")
for i in range(0,nsrcs):
    node_id_to_ranges[i] = i*nedges_per_src, (i+1)*nedges_per_src

range_to_edge_id = source_to_target.create_dataset("range_to_edge_id", (nedges,2), dtype="i")
for i in range(0,nsrcs):
    for j in range(0, nedges_per_src):
        t = (i - j + nsrcs - 1)%nsrcs
        range_to_edge_id[i*nedges_per_src + j] = t*nedges_per_src + j, t*nedges_per_src + j + 1
         
target_to_source = ind.create_group("target_to_source")

node_id_to_ranges = target_to_source.create_dataset("node_id_to_ranges", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    node_id_to_ranges[i] = i, i+1

range_to_edge_id = target_to_source.create_dataset("range_to_edge_id", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    range_to_edge_id[i] = i*nedges_per_tgt, (i+1)*nedges_per_tgt


g0 = pop_i_i.create_group("0")

afferent_id = g0.create_dataset("afferent_section_id", (nedges,), dtype="i")
afferent_pos = g0.create_dataset("afferent_section_pos", (nedges,), dtype="f")
efferent_id = g0.create_dataset("efferent_section_id", (nedges,), dtype="i")
efferent_pos = g0.create_dataset("efferent_section_pos", (nedges,), dtype="f")

for i in range(0,nedges):
    afferent_id[i] = 0
    afferent_pos[i] = 0.5
    efferent_id[i] = 0
    efferent_pos[i] = 0.9

####################################### NOT PARAMETRIZABLE ##################################
nedges = 2000
ntgts = 100
nsrcs = 400
s2t_ratio = nsrcs//ntgts #4
nedges_per_tgt = nedges//ntgts #20
nedges_per_src = nedges//nsrcs #5

edge_group_id = pop_e_i.create_dataset("edge_group_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_id[i] = 0

edge_group_index = pop_e_i.create_dataset("edge_group_index", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_group_index[i] = i

edge_type_id = pop_e_i.create_dataset("edge_type_id", (nedges,), dtype="i")
for i in range(0,nedges):
    edge_type_id[i] = 103

source_node_id = pop_e_i.create_dataset("source_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range (0,nedges_per_src):
        for k in range(0,s2t_ratio):
            src_start = i*s2t_ratio
            source_node_id[i*nedges_per_src*s2t_ratio + j*s2t_ratio + k] = (src_start + k)

target_node_id = pop_e_i.create_dataset("target_node_id", (nedges,), dtype="i")
for i in range(0,ntgts):
    for j in range(0,nedges_per_tgt):
        target_node_id[i*nedges_per_tgt + j] = i


ind = pop_e_i.create_group("indicies")
source_to_target = ind.create_group("source_to_target")

node_id_to_ranges = source_to_target.create_dataset("node_id_to_ranges", (nsrcs,2), dtype="i")
for i in range(0,nsrcs):
    node_id_to_ranges[i] = i*nedges_per_src, (i+1)*nedges_per_src

range_to_edge_id = source_to_target.create_dataset("range_to_edge_id", (nedges,2), dtype="i")
for i in range(0, ntgts):
    for j in range(0, s2t_ratio):
        for k in range(0, nedges_per_src):
            start = j + k*s2t_ratio + i*s2t_ratio*nedges_per_src
            range_to_edge_id[i*s2t_ratio*nedges_per_src + j*nedges_per_src + k] = start, start+1
         
target_to_source = ind.create_group("target_to_source")

node_id_to_ranges = target_to_source.create_dataset("node_id_to_ranges", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    node_id_to_ranges[i] = i, i+1

range_to_edge_id = target_to_source.create_dataset("range_to_edge_id", (ntgts,2), dtype="i")
for i in range(0,ntgts):
    range_to_edge_id[i] = i*nedges_per_tgt, (i+1)*nedges_per_tgt


g0 = pop_e_i.create_group("0")

afferent_id = g0.create_dataset("afferent_section_id", (nedges,), dtype="i")
afferent_pos = g0.create_dataset("afferent_section_pos", (nedges,), dtype="f")
efferent_id = g0.create_dataset("efferent_section_id", (nedges,), dtype="i")
efferent_pos = g0.create_dataset("efferent_section_pos", (nedges,), dtype="f")

for i in range(0,nedges):
    afferent_id[i] = 0
    afferent_pos[i] = 0.5
    efferent_id[i] = 0
    efferent_pos[i] = 0.9

##################################################################################
f.close()
