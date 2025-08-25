from fastapi import FastAPI,Query,HTTPException,File,UploadFile
import os
from reactot.run_model import pred_ts
from types import SimpleNamespace
import tos

app=FastAPI()
ak = os.environ.get('TOS_ACCESS_KEY')
sk = os.environ.get('TOS_SECRET_KEY')
endpoint = os.environ.get('TOS_ENDPOINT')
region = os.environ.get('TOS_REGION')
bucket_name = os.environ.get('BUCKET_NAME')
client = tos.TosClientV2(ak, sk, endpoint, region)


def upload_file(path):
    objects = []
    for file in os.listdir(path):
        if file.endswith('_rxn.xyz') or file.endswith('_ts.xyz'):
            file_path = os.path.join(path, file)
            object_key=f'react-ot/{path}/{file}'
            objects.append(object_key)
            client.put_object_from_file(bucket_name, object_key, file_path)
    return objects




@app.post('/react-ot')
async def main(rxyz:UploadFile = File(...,description='Specify the input file path'),
        pxyz:UploadFile = File(...,description='Specify the product file path'),
        output_path:str=Query(...,description='The output path'),
         ):

    rxyz_path = os.path.join(output_path,rxyz.filename)
    pxyz_path = os.path.join(output_path,pxyz.filename)
    os.makedirs(output_path, exist_ok=True)
    with open(rxyz_path,'wb') as f:
        f.write(await rxyz.read())
    with open(pxyz_path,'wb') as f:
        f.write(await pxyz.read())

    opt = SimpleNamespace(
        rxyz=rxyz_path,
        pxyz=pxyz_path,
        output_path=output_path,
        batch_size=72,
        nfe=100,
        solver='ddpm', # Specify one of ["ddpm", "ei", "ode"]
        checkpoint_path='/root/react-ot/reactot-pretrained.ckpt',
        order=1,
        diz='linear',
        method='midpoint',
        atol=1e-2,
        rtol=1e-2
    )
    pred_ts(rxyz_path, pxyz_path, opt, output_path)
    objects = upload_file(output_path)

    return objects
    

