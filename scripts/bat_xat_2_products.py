from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,math,shutil
import geopandas as gpd, numpy as np, rasterio, matplotlib.pyplot as plt, planetary_computer
from pyproj import CRS,Transformer
from pystac_client import Client
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform,reproject,Resampling
from shapely.geometry import Polygon,mapping
from shapely.ops import transform as shp_transform

R=Path(__file__).resolve().parents[1]; O=R/'outputs/Bat_Xat_2_VN2000'; T=R/'tmp_bx2'; O.mkdir(parents=True,exist_ok=True); T.mkdir(exist_ok=True)
C=[(103.6175505404967,22.56669560881813),(103.6134697596639,22.5724108489074),(103.6117879315856,22.57269932099309),(103.605169761005,22.57231164834358),(103.6003783951688,22.57293696914944),(103.5998200174139,22.57709340242572),(103.599776708036,22.58207063019086),(103.5985592869527,22.58459564574868),(103.5984698472342,22.58593069544057),(103.5998695554187,22.59318052806689),(103.6036342379722,22.59595022747392),(103.6099973382841,22.59060641886939),(103.6157430182426,22.59376613657808),(103.6204077495858,22.59199516331316),(103.6216730258928,22.5891107132241),(103.6229257216092,22.58581263534425),(103.6232235302118,22.58155991621258),(103.6232463791948,22.57861483126316),(103.6228385778411,22.57665505440038),(103.6238774167315,22.57226359430158),(103.6175505404967,22.56669560881813)]
P=Polygon(C); W=CRS.from_epsg(4326); V=CRS.from_epsg(9208); GW=gpd.GeoDataFrame({'name':['Bat xat 2']},geometry=[P],crs=W); GV=GW.to_crs(V)

def geom(crs): return shp_transform(Transformer.from_crs(W,crs,always_xy=True).transform,P)
def crop(url,out,nodata=0):
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
  with rasterio.open(url) as s:
   a,tr=mask(s,[mapping(geom(CRS.from_user_input(s.crs)))],crop=True,nodata=nodata)
   p=s.profile.copy(); p.update(driver='GTiff',height=a.shape[1],width=a.shape[2],transform=tr,count=a.shape[0],nodata=nodata,compress='DEFLATE',tiled=True)
   with rasterio.open(out,'w',**p) as d:d.write(a)
def warp(src,dst,res,nodata,method):
 with rasterio.open(src) as s:
  tr,w,h=calculate_default_transform(s.crs,V,s.width,s.height,*s.bounds,resolution=res); p=s.profile.copy(); p.update(crs=V,transform=tr,width=w,height=h,nodata=nodata,compress='DEFLATE',tiled=True)
  with rasterio.open(dst,'w',**p) as d:
   for i in range(1,s.count+1): reproject(rasterio.band(s,i),rasterio.band(d,i),src_transform=s.transform,src_crs=s.crs,src_nodata=s.nodata,dst_transform=tr,dst_crs=V,dst_nodata=nodata,resampling=method)
def clip(src,dst,nodata):
 with rasterio.open(src) as s:
  a,tr=mask(s,[mapping(GV.geometry.iloc[0])],crop=True,nodata=nodata); p=s.profile.copy();p.update(height=a.shape[1],width=a.shape[2],transform=tr,nodata=nodata,compress='DEFLATE',tiled=True)
  with rasterio.open(dst,'w',**p) as d:d.write(a)

def rgb():
 cat=Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=planetary_computer.sign_inplace); now=datetime.now(timezone.utc)
 it=list(cat.search(collections=['sentinel-2-l2a'],intersects=mapping(P),datetime=f'{(now-timedelta(days=730)).isoformat()}/{now.isoformat()}',query={'eo:cloud_cover':{'lt':45}},max_items=250).items()); newest=max(x.datetime for x in it); recent=[x for x in it if (newest-x.datetime).days<=90]; x=min(recent,key=lambda z:(z.properties.get('eo:cloud_cover',100),-z.datetime.timestamp())); x=planetary_computer.sign(x)
 ps=[]
 for k in ['B04','B03','B02']:
  q=T/f'{k}.tif'; crop(x.assets[k].href,q); ps.append(q)
 with rasterio.open(ps[0]) as s:
  p=s.profile.copy();p.update(count=3); q=T/'rgb.tif'
  with rasterio.open(q,'w',**p) as d:
   for i,f in enumerate(ps,1):
    with rasterio.open(f) as z:d.write(z.read(1),i)
 u=T/'rgb_v.tif'; f=O/'Bat_Xat_2_Sentinel2_RGB_VN2000_EPSG9208_10m.tif';warp(q,u,10,0,Resampling.bilinear);clip(u,f,0)
 with rasterio.open(f) as s:
  a=s.read().astype('float32');m=np.any(a>0,axis=0);b=np.zeros_like(a)
  for i in range(3):
   lo,hi=np.percentile(a[i][m],[2,98]);b[i]=np.clip((a[i]-lo)/(hi-lo+1e-6),0,1)
  im=np.transpose(b,(1,2,0));im[~m]=1;plt.figure(figsize=(9,9));plt.imshow(im);plt.title(f'Sentinel-2 L2A RGB | {x.datetime.date()} | cloud {x.properties.get("eo:cloud_cover",0):.1f}%');plt.axis('off');plt.tight_layout();plt.savefig(O/'Bat_Xat_2_Sentinel2_RGB_preview.png',dpi=220);plt.close()
 return {'item_id':x.id,'date':x.datetime.isoformat(),'cloud':x.properties.get('eo:cloud_cover'),'newest_catalog_date':newest.isoformat()}

def dem():
 url='https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N22_00_E103_00_DEM/Copernicus_DSM_COG_10_N22_00_E103_00_DEM.tif';q=T/'dem.tif';crop(url,q,-9999);u=T/'dem_v.tif';f=O/'Bat_Xat_2_Copernicus_DEM_GLO30_VN2000_EPSG9208_30m.tif';warp(q,u,30,-9999,Resampling.bilinear);clip(u,f,-9999)
 with rasterio.open(f) as s:
  a=s.read(1).astype('float32');v=(a!=s.nodata)&np.isfinite(a)&(a>-1000);fill=np.where(v,a,np.nanmedian(a[v]));dy,dx=np.gradient(fill,abs(s.transform.e),abs(s.transform.a));sl=np.degrees(np.arctan(np.hypot(dx,dy)));asp=np.arctan2(-dx,dy);az=math.radians(315);al=math.radians(45);hs=np.clip(np.sin(al)*np.cos(np.radians(sl))+np.cos(al)*np.sin(np.radians(sl))*np.cos(az-asp),0,1)
  p=s.profile.copy();p.update(dtype='float32',nodata=-9999)
  with rasterio.open(O/'Bat_Xat_2_slope_degrees_VN2000_EPSG9208_30m.tif','w',**p) as d:d.write(np.where(v,sl,-9999).astype('float32'),1)
  p.update(dtype='uint8',nodata=0)
  with rasterio.open(O/'Bat_Xat_2_hillshade_VN2000_EPSG9208_30m.tif','w',**p) as d:d.write(np.where(v,hs*255,0).astype('uint8'),1)
  for arr,cmap,title,name in [(a,'terrain','Copernicus DEM GLO-30 - VN-2000','DEM'),(sl,'magma','Terrain slope (degrees)','slope'),(hs,'gray','Hillshade','hillshade')]:
   plt.figure(figsize=(9,9));plt.imshow(np.ma.masked_where(~v,arr),cmap=cmap);plt.colorbar();plt.title(title);plt.axis('off');plt.tight_layout();plt.savefig(O/f'Bat_Xat_2_{name}_preview.png',dpi=220);plt.close()
  return {'source':url,'min_m':float(np.nanmin(np.where(v,a,np.nan))),'max_m':float(np.nanmax(np.where(v,a,np.nan))),'mean_m':float(np.nanmean(np.where(v,a,np.nan)))}

def main():
 for x in O.iterdir(): shutil.rmtree(x) if x.is_dir() else x.unlink()
 GW.to_file(O/'Bat_Xat_2_boundary_WGS84.geojson',driver='GeoJSON');GV.to_file(O/'Bat_Xat_2_boundary_VN2000_EPSG9208.gpkg',layer='boundary',driver='GPKG')
 sm=rgb();dm=dem();meta={'generated_utc':datetime.now(timezone.utc).isoformat(),'crs':'VN-2000 / TM-3 104-45 (EPSG:9208)','area_km2':float(GV.area.iloc[0]/1e6),'perimeter_km':float(GV.length.iloc[0]/1000),'bounds_wgs84':list(map(float,GW.total_bounds)),'bounds_vn2000':list(map(float,GV.total_bounds)),'sentinel2':sm,'dem':dm};(O/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');(O/'README.txt').write_text('Bat Xat 2 GIS products in VN-2000 / TM-3 104-45 (EPSG:9208).\nSee metadata.json for source dates and statistics.\n',encoding='utf-8');shutil.make_archive(str(R/'Bat_Xat_2_VN2000_products'),'zip',root_dir=O);print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
