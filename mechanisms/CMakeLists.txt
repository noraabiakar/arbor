include(BuildModules.cmake)

# Define catalogues
make_catalogue(
  NAME bbp
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bbp"
  OUTPUT "CAT_BBP_SOURCES"
  MECHS CaDynamics_E2 Ca_HVA Ca_LVAst Ih Im K_Pst K_Tst Nap_Et2 NaTa_t NaTs2_t SK_E2 SKv3_1
  ARBOR "${PROJECT_SOURCE_DIR}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

make_catalogue(
  NAME allen
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/allen"
  OUTPUT "CAT_ALLEN_SOURCES"
  MECHS CaDynamics Ca_HVA Ca_LVA Ih Im Im_v2 K_P K_T Kd Kv2like Kv3_1 NaTa NaTs NaV Nap SK
  ARBOR "${PROJECT_SOURCE_DIR}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

make_catalogue(
  NAME dbbs
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dbbs"
  OUTPUT "CAT_DBBS_SOURCES"
  MECHS Ca CaL13 Cav2_1 Cav2_2 Cav2_3 Cav3_1 Cav3_2 Cav3_3 cdp5 cdp5_CAM cdp5_CAM_GoC cdp5_CR HCN1_golgi HCN1 HCN2 Kca1_1 Kca2_2 Kca3_1 Kir2_3 Km Kv1_1 Kv1_5 Kv2_2 Kv3_3 Kv3_4 Kv4_3 Kv7 Leak_GABA Leak Na_granule_cell_FHF Na_granule_cell Nav1_1 Nav1_6
  ARBOR "${PROJECT_SOURCE_DIR}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

make_catalogue(
  NAME default
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/default"
  OUTPUT "CAT_DEFAULT_SOURCES"
  MECHS exp2syn expsyn expsyn_stdp hh kamt kdrmt nax nernst pas
  ARBOR "${PROJECT_SOURCE_DIR}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

# Join sources
set(arbor_mechanism_sources
  ${CAT_BBP_SOURCES}
  ${CAT_ALLEN_SOURCES}
  ${CAT_DBBS_SOURCES}
  ${CAT_DEFAULT_SOURCES}
  PARENT_SCOPE)

if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${arbor_mechanism_sources} PROPERTIES LANGUAGE CXX)
endif()
