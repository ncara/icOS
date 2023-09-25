# README

The icOS toolbox was installed on the bm07analysis machine (opd07 user)

It can be used as follow

```bash
conda activate online-icOS

cd path-to-data
cp ~/online_spectroscopy/buffer_template.py path-to-data
python buffer_template.py
```

![image](https://github.com/ncara/icOS/assets/77961780/15b1fa8c-0830-49db-acc4-18df39408164)


You just need to modify the timescale if it is not 1 spectrum every second

1. Modify timescale if the recording frquency is not 1 Hz
2. YOu need to modify the shutter_opened value to fit the number of the spectrum at which you opened the xray fast shutter
3. unless you do dose measurement, leave the comented parts out
4. replace “buffer_template” with the name you want the figures to have in the end
5. So far only 650nm is monitored, you can use the replace (ctrl-H) function to set an other wavelength

when you are done, you can either close the terminal or run 

```bash
conda deactivate
```
