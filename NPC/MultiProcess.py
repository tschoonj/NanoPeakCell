import multiprocessing
from NPC.HitFinder import HitFinder
import fabio
import h5py
import NPC.utils as utils
import time
import numpy as np
import random
import os


def dpack(active_areas=None,
          address=None,
          beam_center_x=None,
          beam_center_y=None,
          ccd_image_saturation=None,
          data=None,
          distance=None,
          pixel_size=None,
          saturated_value=None,
          timestamp=None,
          wavelength=None,
          xtal_target=None,
          min_trusted_value=None):


    """XXX Check completeness.  Should fill in sensible defaults."""

    # Must have data.
    if data is None:
        return None

    # Create a time stamp of the current time if none was supplied.
    if timestamp is None:
        timestamp = None

    # For unknown historical reasons, the dictionary must contain both
    # CCD_IMAGE_SATURATION and SATURATED_VALUE items.
    #if ccd_image_saturation is None:
    #    if saturated_value is None:
    #        ccd_image_saturation = cspad_saturated_value
    #    else:
    #        ccd_image_saturation = saturated_value
    #if saturated_value is None:
    #    saturated_value = ccd_image_saturation

    # Use a minimum value if provided for the pixel range
    if min_trusted_value is None:
        #min_trusted_value = cspad_min_trusted_value
        min_trusted_value = None

    # By default, the beam center is the center of the image.  The slow
    # (vertical) and fast (horizontal) axes correspond to x and y,
    # respectively.
    if beam_center_x is None:
        beam_center_x = pixel_size * data.focus()[1] / 2
    if beam_center_y is None:
        beam_center_y = pixel_size * data.focus()[0] / 2

    # By default, the entire detector image is an active area.  There is
    # no sensible default for distance nor wavelength.  XXX But setting
    # wavelength to zero may be disastrous?
    if active_areas is None:
        # XXX Verify order with non-square detector
        active_areas = flex.int((0, 0, data.focus()[0], data.focus()[1]))
    if distance is None:
        distance = 0
    if wavelength is None:
        wavelength = 0

    # The size must match the image dimensions.  The length along the
    # slow (vertical) axis is SIZE1, the length along the fast
    # (horizontal) axis is SIZE2.
    return {'ACTIVE_AREAS': active_areas,
            'BEAM_CENTER_X': beam_center_x,
            'BEAM_CENTER_Y': beam_center_y,
            'CCD_IMAGE_SATURATION': ccd_image_saturation,
            'DATA': data,
            'DETECTOR_ADDRESS': address,
            'DISTANCE': distance,
            'PIXEL_SIZE': pixel_size,
            'SATURATED_VALUE': saturated_value,
            'MIN_TRUSTED_VALUE': min_trusted_value,
            'SIZE1': data.focus()[0],
            'SIZE2': data.focus()[1],
            'TIMESTAMP': timestamp,
            'SEQUENCE_NUMBER': 0,  # XXX Deprecated
            'WAVELENGTH': wavelength,
            'xtal_target': xtal_target}




try:
    #from xfel.cxi.cspad_ana.cspad_tbx import dpack
    #from xfel.command_line.cxi_image2pickle import crop_image_pickle
    from libtbx import easy_pickle
    from scitbx.array_family import flex
    cctbx = True
except ImportError:
    cctbx = False


class FileSentinel(multiprocessing.Process):
    def __init__(self, task_queue, total_queue, options, detector, ai):
        multiprocessing.Process.__init__(self)
        self.kill_received = False
        self.total_queue = total_queue
        self.tasks = task_queue
        self.options = options
        self.get_filenames_mapping = {'SSX': utils.get_filenames}#, 'SFX_SACLA': utils.get_files_sacla}
        self.detector = detector
        self.ai = ai
        self.experiment = self.options['experiment']

        self.live = self.options['live']
        self.all = []
        self.total = 0
        self.chunk = 0

    def load_ssx_queue(self):
        self.det = self.options['detector'].lower()
        if 'eiger' in self.det and 'h5' in self.options['file_extension']:
            self.load_eiger_queue()
        else:
            self.total += len(self.filenames)
            self.total_queue.put(self.total)

            for fname in self.filenames:
                self.tasks.put(fname, block=True, timeout=None)

            if not self.live:
                for i in xrange(self.options['cpus']):
                    self.tasks.put(None)

    def visitor_func(self, name, node):
        if isinstance(node, h5py.Dataset):
            if node.shape[1] * node.shape[2] > 512 * 512: return node.name

    def geth5path(self, fn):
        h5 = h5py.File(fn)
        path = h5.visititems(self.visitor_func)
        type = h5[path].dtype
        ovl = np.iinfo(type).max

        h5.close()
        return (path, ovl, type)

    def load_eiger_queue(self):

        self.h5path, self.overload, self.type = self.geth5path(self.filenames[0])
        tasks = []
        for filename in self.filenames:
            h5 = h5py.File(filename)
            try:
                num_frames, res0, res1 = h5[self.h5path].shape
                self.total += num_frames
                tasks += [(filename, self.h5path, i, self.overload, self.type) for i in xrange(num_frames)]
            except KeyError:
                continue
            h5.close()

        if self.options['randomizer'] not in [False, 0]:
            tasks = [tasks[i] for i in random.sample(xrange(len(tasks)), self.options['randomizer'])]
        self.total_queue.put(self.total)
        self.chunk = max(int(round(float(self.total) / 1000.)), 10)

        for task in tasks:
            self.tasks.put(task, block=True, timeout=None)

        if not self.live:
            for i in xrange(self.options['cpus']):
                self.tasks.put(None)

    def run(self):
      #while not self.kill_received:

        self.filenames = self.get_filenames_mapping[self.experiment](self.options)
        self.load_ssx_queue()
        if self.live:
            while not self.kill_received:
                self.all += self.filenames
                self.filenames = self.get_filenames_mapping[self.experiment](self.options, self.all)
                if self.filenames:
                    if 'eiger' in self.det and 'h5' in self.options['file_extension']:
                        self.load_eiger_queue()
                    else:
                        self.load_ssx_queue()
                time.sleep(10)


class MProcess(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, options, detector, ai):
        multiprocessing.Process.__init__(self)
        self.kill_received = False
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.options = options
        self.detector = detector
        self.ai = ai
        self.signal = True
        self.type = np.float32
        if self.options['roi'] == 'None' :
            self.xmax, self.ymax = self.detector.shape
            self.xmin, self.ymin = 0, 0

        else:
            x1, y1, x2, y2 = self.options['roi'].split()
            self.xmax = max(int(x1), int(x2))
            self.xmin = min(int(x1), int(x2))
            self.ymax = max(int(y1), int(y2))
            self.ymin = min(int(y1), int(y2))

            self.options['ROI_tuple'] = (self.xmin, self.xmax, self.ymin, self.ymax)

        self.HitFinder = HitFinder(self.options, self.detector, self.ai)
        self.data_mapping = {'SSX': self.set_data_ssx, 'SFX_SACLA': self.set_data_sacla}
        self.data = np.zeros(self.detector.shape)
        self.h5 = None
        self.h5_filename = None


    def run(self):

        while not self.kill_received:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                if self.h5 is not None: self.h5.close()
                break
            else:
                result = self.data_mapping[self.options['experiment']](next_task)
            self.result_queue.put(result)
            if result[0] == 1:
                self.set_ssx(next_task)
                self.save_hit()
            self.task_queue.task_done()
        return

    def set_data_ssx(self,filename):
        if 'eiger' in self.options['detector'].lower() and 'h5' in self.options['file_extension']:
            return self.set_data_eiger(filename)
        else:
            try:
              self.img = fabio.open(filename)
              self.HitFinder.data[:] = self.img.data[self.xmin:self.xmax,self.ymin:self.ymax]

            except AssertionError:
              time.sleep(1)
              self.img = fabio.open(filename)
              self.HitFinder.data[:] = fabio.open(filename).data[self.xmin:self.xmax,self.ymin:self.ymax]

            return self.HitFinder.get_hit(filename)

    def set_data_eiger(self,task):
        filename, self.group, self.index, self.ovl, type = task
        if self.h5 == None:
            self.h5 = h5py.File(filename,'r')
            self.h5_filename = filename
        if filename != self.h5_filename:
            self.h5.close()
            self.h5 = h5py.File(filename,'r')
            self.h5_filename = filename
        self.HitFinder.data[:] = self.h5[self.group][self.index,self.xmin:self.xmax,self.ymin:self.ymax]
        self.HitFinder.data[ self.HitFinder.data >= self.ovl] = 0
        return self.HitFinder.get_hit(task)

    def set_ssx(self, fname):
        if 'eiger' in self.options['detector'].lower() and 'h5' in self.options['file_extension']:
            self.filename, group, index, ovl, self.type = fname
            fileout = self.filename.split('.h5')[0]
            fileout = os.path.basename(fileout)
            self.root = "%s_%s"%(fileout, str(index).zfill(6))

        else:
            self.root = os.path.basename(fname)
            self.root, self.extension = os.path.splitext(self.root)

    def save_hit(self):

        #self.set_ssx()
        self.result_folder = self.options['output_directory']
        self.num = self.options['num']

        if self.options['roi'].lower() is not 'none':
            if 'eiger' in self.options['detector'].lower() and 'h5' in self.options['file_extension']:
                self.data = self.h5[self.group][self.index,::]
                self.data[self.data >= self.ovl] = 0
            else: self.data = self.img.data


        # Conversion to edf
        if 'edf' in self.options['output_formats']:
            OutputFileName = os.path.join(self.result_folder, 'EDF_%s' % self.num.zfill(3), "%s.edf" % self.root)
            edfout = fabio.edfimage.edfimage(data=self.data.astype(np.float32))
            edfout.write(OutputFileName)

        if 'cbf' in self.options['output_formats']:
            OutputFileName = os.path.join(self.result_folder, 'CBF_%s' % self.num.zfill(3), "%s.cbf" % self.root)
            cbfout = fabio.cbfimage.cbfimage(data=self.data.astype(np.float32))
            cbfout.write(OutputFileName)

        # Conversion to H5
        if 'hdf5' in self.options['output_formats']:

            OutputFileName = os.path.join(self.result_folder,
                                          'HDF5_%s_%s' % (self.options['filename_root'], self.num.zfill(3)),
                                          "%s.h5" % self.root)
            OutputFile = h5py.File(OutputFileName, 'w')
            OutputFile.create_dataset("data", data=self.data, compression="gzip", dtype=self.type)
            if self.options['bragg_search']:
                OutputFile.create_dataset("processing/hitfinder/peakinfo", data=self.peaks.astype(np.int))
            OutputFile.close()

        # Conversion to Pickle
        if cctbx and 'pickles' in self.options['output_formats']:
            # def get_ovl(det):
            if 'pilatus' in self.detector.name.lower(): ovl = 1048500
            if 'eiger' in self.detector.name.lower(): ovl = self.ovl
            pixels = flex.int(self.data.astype(np.int32))
            pixel_size = self.detector.pixel1
            data = dpack(data=pixels,
                         distance=self.options['distance'],
                         pixel_size=pixel_size,
                         wavelength=self.options['wavelength'],
                         beam_center_x=self.options['beam_y'] * pixel_size,
                         beam_center_y=self.options['beam_x'] * pixel_size,
                         ccd_image_saturation= ovl,
                         saturated_value= ovl)
            #data = crop_image_pickle(data)
            OutputFileName = os.path.join(self.result_folder,
                                          'PICKLES_%s_%s' %(self.options['filename_root'],
                                                            self.num.zfill(3)),
                                          "%s.pickle" % self.root)
            easy_pickle.dump(OutputFileName, data)

    def set_data_sacla(self,task):
        filename, run, tag = task
        if self.h5 == None:
            self.h5 = h5py.File(filename)
            self.h5_filename = filename
        if filename != self.h5_filename:
            self.h5.close()
            self.h5 = h5py.File(filename)
            self.h5_filename = filename
        self.HitFinder.data[:] = self.h5['%s/detector_2d_assembled_1/tag_%s/detector_data'%(run,tag)][self.xmin:self.xmax,self.ymin:self.ymax]
        return self.HitFinder.get_hit((run,tag))


    def OnStop(self):
        try:
            self.signal = False
        except:
            return

