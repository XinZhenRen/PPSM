#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
##
#
python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=SUN --data-root=./SUN/
python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict-prompt.py --model=PPSN --p=1 --train-dataset=Kvasir --lastName=epoch20_mean0.9806592667102814_devicecuda_p1 --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=Kvasir --lastName=epoch20_mean0.9806592667102814_devicecuda_p1 --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=Kvasir --lastName=epoch20_mean0.9806592667102814_devicecuda_p1 --test-dataset=SUN --data-root=./SUN/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=Kvasir --lastName=epoch20_mean0.9806592667102814_devicecuda_p1 --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=CVC --lastName=epoch19_mean0.9912799342733914_devicecuda_p1 --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=CVC --lastName=epoch19_mean0.9912799342733914_devicecuda_p1 --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=CVC --lastName=epoch19_mean0.9912799342733914_devicecuda_p1 --test-dataset=SUN --data-root=./SUN/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=CVC --lastName=epoch19_mean0.9912799342733914_devicecuda_p1 --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=SUN --lastName=epoch16_mean0.993876799925811_devicecuda_p1 --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=SUN --lastName=epoch16_mean0.993876799925811_devicecuda_p1 --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=SUN --lastName=epoch16_mean0.993876799925811_devicecuda_p1 --test-dataset=SUN --data-root=./SUN/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=SUN --lastName=epoch16_mean0.993876799925811_devicecuda_p1 --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=polyGen --lastName=epoch16_mean1.1570899135413322_devicecuda_p1 --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=polyGen --lastName=epoch16_mean1.1570899135413322_devicecuda_p1 --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=polyGen --lastName=epoch16_mean1.1570899135413322_devicecuda_p1 --test-dataset=SUN --data-root=./SUN/
#python predict-prompt.py --model=PPSN --p=1 --train-dataset=polyGen --lastName=epoch16_mean1.1570899135413322_devicecuda_p1 --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
