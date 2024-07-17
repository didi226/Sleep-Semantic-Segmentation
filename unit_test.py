import unittest
import mne
import yasa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wrap_sssm.detection import sleep_event_detect
sns.set(font_scale=1.2)
class MyTestCase(unittest.TestCase):
    def test_summary(self):
        raw = mne.io.read_raw_edf('./notebookes/SC4001E0-PSG.edf', preload=True)
        raw.filter(0.1, 40)
        data = raw.get_data(['EEG Fpz-Cz'], units="uV")
        print(data[0,:].shape)
        sf = 100
        sp = sleep_event_detect(data[:,:50000], sf)
        # sp.calculate_feature()
        sp.summary()
        # print(sp.summary())
        # figure = sp.plot_average()
        # plt.show()
        # sp.plot_detection()
        # plt.show()




if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        [MyTestCase('test_summary')])  # test_net_eegnet_TR_crosssub  test_psd test_insub_classify
    runner = unittest.TextTestRunner()  # 通过unittest自带的TextTestRunner方法
    runner.run(suite)