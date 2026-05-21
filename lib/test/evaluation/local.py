from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/sqh/lihao/RAGTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/sqh/lihao/RAGTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/sqh/lihao/RAGTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/sqh/lihao/RAGTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/sqh/lihao/RAGTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/sqh/lihao/RAGTrack/data/lasot'
    settings.network_path = '/home/sqh/lihao/RAGTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/sqh/lihao/RAGTrack/data/nfs'
    settings.otb_path = '/home/sqh/lihao/RAGTrack/data/otb'
    settings.prj_dir = '/home/sqh/lihao/RAGTrack'
    settings.result_plot_path = '/home/sqh/lihao/RAGTrack/output/test/result_plots'
    settings.results_path = '/home/sqh/lihao/RAGTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/sqh/lihao/RAGTrack/output'
    settings.segmentation_path = '/home/sqh/lihao/RAGTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/sqh/lihao/RAGTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/sqh/lihao/RAGTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/sqh/lihao/RAGTrack/data/trackingnet'
    settings.uav_path = '/home/sqh/lihao/RAGTrack/data/uav'
    settings.vot18_path = '/home/sqh/lihao/RAGTrack/data/vot2018'
    settings.vot22_path = '/home/sqh/lihao/RAGTrack/data/vot2022'
    settings.vot_path = '/home/sqh/lihao/RAGTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

