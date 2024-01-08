import Area from "features.js";
Map.addLayer(Area,{}, 'Area')

// Merge the three geometry layers into a single FeatureCollection.
var trainingData = builtuparea.merge(water).merge(vegetation).merge(agriculture);

var samples=ee.FeatureCollection(trainingData);//here add your training samples


function maskL8sr(image)
{
  var timeStart = image.get('system:time_start');
  var srImageList = ee.ImageCollection(' LANDSAT/LC08/C01/T1_SR')
                  .filterMetadata('system:time_start','equals',timeStart)
                  .toList(5);
  var cloudShadowBitMask = ee.Number(2).pow(3).int();
  var cloudsBitMask = ee.Number(2).pow(5).int();



  var qa = image.select('pixel_qa');


  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0));


  return image.updateMask(mask);
}


function maskL7sr(image) 
{
  var timeStart = image.get('system:time_start');
  var srImageList = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
                      .filterMetadata('system:time_start','equals',timeStart)
                      .toList(5);
                       
  var qa = image.select('pixel_qa');

  var cloud = qa.bitwiseAnd(1 << 5)
          .and(qa.bitwiseAnd(1 << 7))
          .or(qa.bitwiseAnd(1 << 3))
  var mask = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask);
}

function maskL5sr(image) 
{
  var timeStart = image.get('system:time_start');
  var srImageList = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
                      .filterMetadata('system:time_start','equals',timeStart)
                      .toList(5);
  var qa = image.select('pixel_qa');
  var cloud = qa.bitwiseAnd(1 << 5)
          .and(qa.bitwiseAnd(1 << 7))
          .or(qa.bitwiseAnd(1 << 3))
  var mask = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask);
}

function ND_VI(image,b1,b2,bName)
{
  var VI = image.normalizedDifference([b1,b2]).rename(bName);
  return VI.updateMask(VI.gt(-1).and(VI.lt(1)));
}

function funEVI(image,B1,B2,B3)
{
 
	var VI = image.expression('2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
    {
      blue: image.select(B1).multiply(0.0001),   
      red:  image.select(B2).multiply(0.0001),   
      nir:  image.select(B3).multiply(0.0001)
    }).rename('EVI');
  return VI.updateMask(VI.gt(-1).and(VI.lt(1)));

}

function addLandsatVIs(img)
{
  var NDVI = ND_VI(img,'B4','B3','NDVI');
  var EVI = funEVI(img,'B1','B3','B4');
  var LSWI = ND_VI(img,'B4','B5','LSWI');
  return img.addBands(NDVI).addBands(EVI).addBands(LSWI);
}

var collection_L8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    .filterBounds(Area)
    .filterDate('2013-01-01','2020-12-31')
    .map(maskL8sr)
    .select( 
      ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa']
    ,['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
    .map(addLandsatVIs);

var collection_L7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
    .filterBounds(Area)
    .filterDate('1999-01-01','2020-01-01')
    .map(maskL7sr)
    .select( 
      ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
    .map(addLandsatVIs);

var collection_L5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    .filterBounds(Area)
    .filterDate('1988-01-01','2012-01-01')
    .map(maskL5sr)
    .select( 
      ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
    .map(addLandsatVIs);

var collection=ee.ImageCollection(collection_L5.merge(collection_L7).merge(collection_L8));


var LandSatCollection=collection.filterDate('2020-01-01','2020-12-31').filterBounds(Area)

Map.addLayer(LandSatCollection.median().clip(Area),{min:0,max:3000,bands:['B4','B3','B2']},'Landsat2020')
Map.centerObject(Area,7)
var sampleData = samples.randomColumn('random');
var sample_training = sampleData.filter(ee.Filter.lte("random", 0.8)); 
var sample_validate  = sampleData.filter(ee.Filter.gt("random", 0.8));



var data=ee.Image.cat(LandSatCollection.median())


var training = data.sampleRegions({
  collection: sample_training, 
  properties: ["landcover"], 
  scale: 30
});

var validation = data.sampleRegions({
  collection: sample_validate, 
  properties: ["landcover"], 
  scale: 30
});


var classifier = ee.Classifier.smileRandomForest(40)
    .train({
      features: training, 
      classProperty: 'landcover', 
      inputProperties: data.bandNames()
    });




var Classified_2020 = data.classify(classifier);

Map.addLayer(Classified_2020.clip(Area), {min:0,max:4,palette:['red','darkslategrey','blue','green','gold']},  'Classified_2020');


var validated = validation.classify(classifier);
var testAccuracy = validated.errorMatrix('landcover', 'classification');
var accuracy = testAccuracy.accuracy();
var userAccuracy = testAccuracy.consumersAccuracy();
var producersAccuracy = testAccuracy.producersAccuracy();
var kappa = testAccuracy.kappa();
print('Validation error matrix:', testAccuracy);
print('Validation overall accuracy:', accuracy);
print('User acc:', userAccuracy);
print('Prod acc:', producersAccuracy);
print('Kappa:', kappa);


// //Finding Pixel Area of landsat image

var class_areas = ee.Image.pixelArea().addBands(Classified_2020)
  .reduceRegion({
    reducer: ee.Reducer.sum().group({
      groupField: 1,
      groupName: 'landcover',
    }),
    geometry: Area,
    scale: 30,  // sample the geometry at 1m intervals
    maxPixels: 1e10
  }).get('groups');
  
  print(class_areas);
  
  Export.image.toDrive({
  image: Classified_2020,
  description: 'LULC2020',
  folder: "Thesis LULC Data Year Wise Without Slum",
  scale: 30,
  crs: 'EPSG:3106',
  region: Area,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});
 