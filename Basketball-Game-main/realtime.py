import argparse
import logging

import sys

import pyqtgraph as pg
import numpy as np
from PyQt5 import QtWidgets
import PyQt5.QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer 
import time
from pynput.keyboard import Key, Controller
#import pydirectinput
import matplotlib 
import matplotlib.pyplot as plt


#import pyqtgraph as pg
#from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
#from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
#from PyQt5.QtWidgets import QApplication

class Graph:
    def __init__(self, board_shim):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.fhandle = open("silly.txt", "w")

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(
            title='BrainFlow Plot', size=(800, 600))
        self.win.show()

        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()
        self.jawclench = 0

        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)

        sys.exit(self.app.exec_())

    def _init_pens(self):
        self.pens = list()
        self.brushes = list()
        colors = ['#A54E4E', '#A473B6', '#5B45A4', '#2079D2',
                  '#32B798', '#2FA537', '#9DA52F', '#A57E2F', '#A53B2F']
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot(pen=self.pens[i % len(self.pens)])
            # curve.setDownsampling(auto=True, method='mean', ds=3)
            self.curves.append(curve)

    def _init_psd(self):
        self.psd_plot = self.win.addPlot(
            row=0, col=1, rowspan=len(self.exg_channels) // 2)
        self.psd_plot.showAxis('left', False)
        self.psd_plot.setMenuEnabled('left', False)
        self.psd_plot.setTitle('PSD Plot')
        self.psd_plot.setLogMode(False, True)
        self.psd_curves = list()
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        for i in range(len(self.exg_channels)):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            psd_curve.setDownsampling(auto=True, method='mean', ds=3)
            self.psd_curves.append(psd_curve)

    def _init_band_plot(self):
        self.band_plot = self.win.addPlot(
            row=len(self.exg_channels) // 2, col=1, rowspan=len(self.exg_channels) // 2)
        self.band_plot.showAxis('left', False)
        self.band_plot.setMenuEnabled('left', False)
        self.band_plot.showAxis('bottom', False)
        self.band_plot.setMenuEnabled('bottom', False)
        self.band_plot.setTitle('BandPower Plot')
        y = [0, 0, 0, 0, 0]
        x = [1, 2, 3, 4, 5]
        self.band_bar = pg.BarGraphItem(
            x=x, height=y, width=0.8, pen=self.pens[0], brush=self.brushes[0])
        self.band_plot.addItem(self.band_bar)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        isFocused = False
        isJawClench = False
        keyboard = Controller()
        
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())
            if data.shape[1] > self.psd_size:
                # plot psd
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2,
                                                    self.sampling_rate,
                                                    WindowOperations.BLACKMAN_HARRIS.value)
                lim = min(70, len(psd_data[0]))
                self.psd_curves[count].setData(
                    psd_data[1][0:lim].tolist(), psd_data[0][0:lim].tolist())
                # plot bands
                avg_bands[0] = avg_bands[0] + \
                    DataFilter.get_band_power(psd_data, 2.0, 4.0)
                self.fhandle.write(str(avg_bands[0]) + ", ")
                avg_bands[1] = avg_bands[1] + \
                    DataFilter.get_band_power(psd_data, 4.0, 8.0)
                self.fhandle.write(str(avg_bands[1]) + ", ")
                avg_bands[2] = avg_bands[2] + \
                    DataFilter.get_band_power(psd_data, 8.0,  13.0)
                self.fhandle.write(str(avg_bands[2]) + ", ") 
                avg_bands[3] = avg_bands[3] + \
                    DataFilter.get_band_power(psd_data, 13.0, 30.0)
                self.fhandle.write(str(avg_bands[3]) + ", ")
                avg_bands[4] = avg_bands[4] + \
                    DataFilter.get_band_power(psd_data, 30.0, 50.0)
                self.fhandle.write(str(avg_bands[4]) + ", ")
                self.fhandle.write("\t")
                if(isFocused):
				    #add JC code here
                    jawclenches = []
                    
                    powers = []
                        
                        
                    chanstudy = data[channel]
                    #print(chanstudy)
                    seconds = 0
                    for j in range(len(chanstudy)):
                        if j % 624 != 0:                                                                                                                                                                             #1248 != 0:              
                                continue
                        binned = chanstudy[j:(j+1248)]

                        signal = np.array(binned)
                        power = np.mean(np.square(signal))
                        #print("Power of the signal for", str(i), 'to', str(i+5) + ':', power)
                        powers.append(power)

                    #powers = np.delete(powers, 0)
                    #print(powers)
                    # Calculate the five-number summary using NumPy
                    min_val = np.min(powers)
                    q1_val = np.percentile(powers, 25)
                    median_val = np.percentile(powers, 50)
                    q3_val = np.percentile(powers, 75)
                    max_val = np.max(powers)





                    # Display the five-number summary
                    print("Channel " + str(count) + ':')
                    print("Minimum:", min_val)
                    print("1st Quartile (Q1):", q1_val)
                    print("Median (Q2):", median_val)
                    print("3rd Quartile (Q3):", q3_val)
                    print("Maximum:", max_val)
                        
                        # Create a box plot
                    #fig, ax = plt.subplots()
                    #ax.boxplot(powers, vert=False)

                        # Set labels for the five-number summary
                    #ax.scatter([min_val, q1_val, median_val, q3_val, max_val], [1, 1, 1, 1, 1], c='red', zorder=3)
                    #ax.text(min_val, 1.15, f"{min_val}", ha='center', va='center')
                    #ax.text(q1_val, 1.15, f"{q1_val}", ha='center', va='center')
                    #ax.text(median_val, 1.15, f"{median_val}", ha='center', va='center')
                    #ax.text(q3_val, 1.15, f"{q3_val}", ha='center', va='center')
                    #ax.text(max_val, 1.15, f"{max_val}", ha='center', va='center')

                    # Set x-axis label
                    #ax.set_xlabel('Values')

                    # Remove y-axis ticks and labels
                    #ax.set_yticks([])
                    #ax.set_yticklabels([])

                    # Set title and display the plot
                    #plt.title('Five-Number Summary')
                    #plt.show()
                        
                        
                    #print(chan1)
                    #eeg_data.columns = ['Channel 1', 'B', 'C'] 
                    signal = np.array(chanstudy)
                    power = np.mean(np.square(signal))
                    #print("Power of the signal:", power)



                    
                    for i in range(len(powers)):
                        if 1000 < powers[i] < 1400 and time.time() > seconds + 1    :
                            
                            isJawclench = True
                            self.jawclench = self.jawclench + 1      
                            time.sleep(5)  #added on wednesday                                                                                                                                                                                                                                                                    
                            keyboard.press(' ')
                            keyboard.release(' ')
                            isJawclench = False #added on wednesday
                            seconds = time.time()
                            break
                        
                        print(self.jawclench)
                        #print(jawclench)                                                                
                    #jawclenches.append(jawclench)    
                        #if (isJawClench):
                            #self.band_plot.setTitle('Space')
                            #keyboard.press(' ')
                            #keyboard.release(' ')
                        #pydirectinput.press('space')
                else:
                    if (avg_bands[2] < 3500 and avg_bands[3] < 2000):
                        isFocused = True
                
                    
        avg_bands = [int(x * 100 / len(self.exg_channels)) for x in avg_bands]
        self.band_bar.setOpts(height=avg_bands)

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int,
                        help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str,
                        help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str,
                        help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str,
                        help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str,
                        help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str,
                        help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str,
                        help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
