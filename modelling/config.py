# RAW DATA

## UAV data
timbertiere_path = '../donnees_terrain/donnees_brutes_initiales/timbertiere/'
timbertiere_spring_MS_path = timbertiere_path + 'Strange_Ortho_MS_mission1/2024-04-29_Timbertiere_MS_DualMX_ortho_8cm.tif'
timbertiere_spring_dsm_path = timbertiere_path + '20240626_Timbertiere_LiDAR/2024-06-26_Timbertiere_dsm.tif'
timbertiere_spring_dtm_path = timbertiere_path + '20240626_Timbertiere_LiDAR/2024-06-26_Timbertiere_dtm.tif'
timbertiere_spring_dhm_path = timbertiere_path + '20240626_Timbertiere_LiDAR/2024-06-26_Timbertiere_dhm.tif'

fao_path = '../donnees_terrain/donnees_brutes_initiales/fao/'
fao_spring_MS_path = fao_path + "2024-04-18_Le-Fao_livrables_20250108/2_raster/ortho_MS/2024-04-18_Fao_MS_5bandes_8cm.tif"
fao_spring_dsm_path = fao_path + '2024-04-18_Le-Fao_livrables_20250108/2_raster/elevation/2024-04-18_Le_Fao_LiDAR_DSM_10cm_one_band.tif'
fao_spring_dtm_path = fao_path + '2024-04-18_Le-Fao_livrables_20250108/2_raster/elevation/2024-04-18_Le_Fao_LiDAR_DTM_10cm_one_band.tif'
fao_spring_dhm_path = fao_path + '2024-04-18_Le-Fao_livrables_20250108/2_raster/elevation/2024-04-18_Le_Fao_LiDAR_DHM_10cm_one_band.tif'

roudoudour_path = '../donnees_terrain/donnees_brutes_initiales/roudoudour/'
roudoudour_spring_MS_path = roudoudour_path + "2024-04-18_Roudoudour_delivery/Data/Multi_Spectral_Ortho/2024-04-18_Roudoudour_MS_10bandes_2154_10cm.tif"
roudoudour_spring_dsm_path = roudoudour_path + '2024-04-18_Roudoudour_delivery/Data/Digital_Elevation_Model/2024-04-18_Roudoudour_LiDAR_2154_IGN69_DSM_10cm.tif'
roudoudour_spring_dtm_path = roudoudour_path + '2024-04-18_Roudoudour_delivery/Data/Digital_Elevation_Model/2024-04-18_Roudoudour_LiDAR_2154_IGN69_DTM_10cm.tif'
roudoudour_spring_dhm_path = roudoudour_path + '2024-04-18_Roudoudour_delivery/Data/Digital_Elevation_Model/2024-04-18_Roudoudour_LiDAR_2154_DHM_10cm.tif'

cisse_path = '../donnees_terrain/donnees_brutes_initiales/cisse/'
cisse_spring_MS_path = cisse_path + "2024-04-24_Cisse_livrables/2024-04-24_Cisse_MS.tif"
cisse_spring_dsm_path = cisse_path + '2024-04-24_Cisse_livrables/DEM/2024-04-24_Cisse_dsm.tif'
cisse_spring_dtm_path = cisse_path + '2024-04-24_Cisse_livrables/DEM/2024-04-24_Cisse_dtm.tif'
cisse_spring_dhm_path = cisse_path + '2024-04-24_Cisse_livrables/DEM/2024-04-24_Cisse_dhm.tif'

louroux_path = '../donnees_terrain/donnees_brutes_initiales/louroux/'
louroux_spring_MS_path = louroux_path + "2024-04-24-Louroux/2024-04-24_Louroux_DualMX_ortho_8cm.tif"
louroux_spring_dsm_path = louroux_path + '2024-04-24-Louroux/2024-04-24_Louroux_DEM/YS-20240424-075215_Louroux_dsm.tif'
louroux_spring_dtm_path = louroux_path + '2024-04-24-Louroux/2024-04-24_Louroux_DEM/YS-20240424-075215_Louroux_dtm.tif'
louroux_spring_dhm_path = louroux_path + '2024-04-24-Louroux/2024-04-24_Louroux_DEM/YS-20240424-075215_Louroux_dhm.tif'

## Mapping of water zones
water_path = '../donnees_terrain/donnees_brutes_initiales/Cartographie réseau hydro.gpkg'

# Trap results
result_stickies_path = "../donnees_terrain/donnees_brutes_initiales/Results_stickies/"
result_stickies_spring_path = result_stickies_path + "spring/"
result_stickies_summer_path = result_stickies_path + "summer/"


# MODIFIED DATA

## Train test val split
train_test_split_output_dir = '../../donnees_terrain/donnees_modifiees/train_test_split/'

train_split_output_dir = train_test_split_output_dir + "train/"
train_patches_output_dir = train_test_split_output_dir + "train_patches/"
timbertiere_spring_train_patches_dir = train_patches_output_dir + "timbertiere_spring/"
timbertiere_summer_train_patches_dir = train_patches_output_dir + "timbertiere_summer/"

test_split_output_dir = train_test_split_output_dir + "test/"
test_patches_output_dir = train_test_split_output_dir + "test_patches/"
timbertiere_spring_test_patches_dir = test_patches_output_dir + "timbertiere_spring/"
timbertiere_summer_test_patches_dir = test_patches_output_dir + "timbertiere_summer/"

val_split_output_dir = train_test_split_output_dir + "val/"
val_patches_output_dir = train_test_split_output_dir + "val_patches/"
timbertiere_spring_val_patches_dir = val_patches_output_dir + "timbertiere_spring/"
timbertiere_summer_val_patches_dir = val_patches_output_dir + "timbertiere_summer/"

## Modified sticky traps' results based on train test val split
split_result_stickies = "../../donnees_terrain/donnees_modifiees/split_result_stickies/random_forest/"

## generated maps
output_map_path = "../donnees_terrain/donnees_modifiees/output_maps/"
timbertiere_spring_water_path = output_map_path + "water_distance_map_timbertiere.tif"
fao_spring_water_path = output_map_path + "water_distance_map_fao.tif"
roudoudour_spring_water_path = output_map_path + "water_distance_map_roudoudour.tif"
louroux_spring_water_path = output_map_path + 'water_distance_map_louroux.tif'
cisse_spring_water_path = output_map_path + 'water_distance_map_cisse.tif'

timbertiere_spring_water_map_path = output_map_path + f"water_maps/spring/timbertiere_water_map.tif"
fao_spring_water_map_path = output_map_path + f"water_maps/spring/fao_water_map.tif"
roudoudour_spring_water_map_path = output_map_path + f"water_maps/spring/roudoudour_water_map.tif"
louroux_spring_water_map_path = output_map_path + f"water_maps/spring/louroux_water_map.tif"
cisse_spring_water_map_path = output_map_path + f"water_maps/spring/cisse_water_map.tif"

# Gaussian optimal transport
ot_path = "../donnees_terrain/donnees_modifiees/MS_rasters_ot/"
timbertiere_ot = timbertiere_spring_MS_path
louroux_ot = ot_path + "louroux_MS_5bandes_adapted_to_timbertiere_resampled_32cm_gaussian.tif"
cisse_ot = ot_path + "cisse_MS_5bandes_adapted_to_timbertiere_resampled_32cm_gaussian.tif"
fao_ot = ot_path + "fao_MS_5bandes_adapted_to_timbertiere_resampled_32cm_gaussian.tif"
roudoudour_ot = ot_path + "roudoudour_MS_5bandes_adapted_to_timbertiere_resampled_32cm_gaussian.tif"

## Labels
labels_path = "labels.csv"

## Pseudo-patches
pseudo_patch_path = "../donnees_terrain/donnees_modifiees/pseudo_patches/"

# Min Max Scaling
min_max_scaling_dict = {'spring': {'roudoudour': {'image': {'all': {'min': 0.0, 'max': 0.73025763},
    'b0': {'min': 0.0, 'max': 0.12936406},
    'b1': {'min': 0.0, 'max': 0.1846306},
    'b2': {'min': 0.0, 'max': 0.19454965},
    'b3': {'min': 0.0, 'max': 0.22015174},
    'b4': {'min': 0.0, 'max': 0.2672342},
    'b5': {'min': 0.0, 'max': 0.3147686},
    'b6': {'min': 0.0, 'max': 0.35447654},
    'b7': {'min': 0.0, 'max': 0.39116704},
    'b8': {'min': 0.0, 'max': 0.5368871},
    'b9': {'min': 0.0, 'max': 0.73025763}},
   'dtm': {'b0': {'min': 220.689, 'max': 234.52}},
   'water': {'b0': {'min': 0.0, 'max': 3221.4397}},
   'dsm': {'b0': {'min': 220.689, 'max': 251.755}}},
  'fao': {'image': {'all': {'min': 0.0, 'max': 0.8991503},
    'b0': {'min': 0.0, 'max': 0.25521055},
    'b1': {'min': 0.0, 'max': 0.32406744},
    'b2': {'min': 0.0, 'max': 0.47137824},
    'b3': {'min': 0.0, 'max': 0.58070445},
    'b4': {'min': 0.0, 'max': 0.8991503}},
   'water': {'b0': {'min': 0.0, 'max': 4595.834}},
   'dsm': {'b0': {'min': 279.534, 'max': 337.908}},
   'dtm': {'b0': {'min': 279.534, 'max': 318.591}}},
  'timbertiere': {'image': {'all': {'min': 0.0, 'max': 1.5025127},
    'b0': {'min': 0.0, 'max': 0.37456056},
    'b1': {'min': 0.0, 'max': 0.46032944},
    'b2': {'min': 0.0, 'max': 0.575613},
    'b3': {'min': 0.0, 'max': 0.67549896},
    'b4': {'min': 0.0, 'max': 0.6211662},
    'b5': {'min': 0.0, 'max': 0.5674679},
    'b6': {'min': 0.0, 'max': 0.8398119},
    'b7': {'min': 0.0, 'max': 0.87591517},
    'b8': {'min': 0.0, 'max': 1.002895},
    'b9': {'min': 0.0, 'max': 1.5025127}},
   'dsm': {'b0': {'min': 135.346, 'max': 177.522}},
   'dtm': {'b0': {'min': 135.127, 'max': 170.939}},
   'water': {'b0': {'min': 0.0, 'max': 3087.697}}},
  'louroux': {'water': {'b0': {'min': 0.0, 'max': 9787.262}},
   'dsm': {'b0': {'min': 145.646, 'max': 187.412}},
   'dtm': {'b0': {'min': 145.645, 'max': 169.915}},
   'image': {'all': {'min': 0.0, 'max': 1.0893089},
    'b0': {'min': 0.0, 'max': 0.55654705},
    'b1': {'min': 0.0, 'max': 0.5837494},
    'b2': {'min': 0.0, 'max': 0.66377246},
    'b3': {'min': 0.0, 'max': 0.6849422},
    'b4': {'min': 0.0, 'max': 0.6784863},
    'b5': {'min': 0.0, 'max': 0.6874557},
    'b6': {'min': 0.0, 'max': 0.6831611},
    'b7': {'min': 0.0, 'max': 0.6847973},
    'b8': {'min': 0.0, 'max': 0.924226},
    'b9': {'min': 0.0, 'max': 1.0893089}}},
  'cisse': {'image': {'all': {'min': 0.0, 'max': 1.0582517},
    'b0': {'min': 0.0, 'max': 0.49641982},
    'b1': {'min': 0.0, 'max': 0.53288895},
    'b2': {'min': 0.0, 'max': 0.51810515},
    'b3': {'min': 0.0, 'max': 0.5547561},
    'b4': {'min': 0.0, 'max': 0.63298076},
    'b5': {'min': 0.0, 'max': 0.64653635},
    'b6': {'min': 0.0, 'max': 0.68079394},
    'b7': {'min': 0.0, 'max': 0.69988877},
    'b8': {'min': 0.0, 'max': 0.89285576},
    'b9': {'min': 0.0, 'max': 1.0582517}},
   'dsm': {'b0': {'min': 137.78, 'max': 184.347}},
   'dtm': {'b0': {'min': 137.78, 'max': 165.02}},
   'water': {'b0': {'min': 0.0, 'max': 8663.143}}}}}
