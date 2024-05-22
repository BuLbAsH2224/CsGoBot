let {mouse, screen, Point, straightTo, getActiveWindow, Button} = require("@nut-tree/nut-js")
const fs = require("fs")
let tf = require("@tensorflow/tfjs-node")
const cocoSsd = require('@tensorflow-models/coco-ssd');
let model = null;
let fps = 0;
mouse.config.mouseSpeed = 100000;

async function loadModel(){model = await cocoSsd.load();}


async function Aim(predictions,windowSize){    
  let persons = predictions.filter(item => item.class === "person");
  let center = await mouse.getPosition()
  let person = persons.reduce((closest, item) => {
    let distance = Math.abs((windowSize.left + item.bbox[0] + item.bbox[2] / 2) - center.x);
    return distance < closest.distance ? {distance, item} : closest;
}, {distance: Infinity, item: null}).item;

          if(person != undefined){
            let Target = new Point(person.bbox[0] + (person.bbox[2] / 2) + windowSize.left,person.bbox[1]  + windowSize.top)
           await mouse.move(straightTo(Target))
            await mouse.click(Button.LEFT);
          }
          fps++;
           screenshot()
}

async function screenshot() {
  let activeWindow = await getActiveWindow();
  let region = await activeWindow.region;
  let image = await screen.grabRegion(region);
  let buffer = image.data;
  const tensor = tf.tensor3d(buffer, [region.height, region.width, 4], 'int32');
  let pngBuffer = await tf.node.encodePng(tensor);
  let img = tf.node.decodeImage(pngBuffer, 3);
  
 model.detect(img).then(pr=>{Aim(pr,region)})
 
}


//setTimeout(()=>screenshot(),1000)
loadModel().then(() => {screenshot()})
setInterval(() => {console.log(fps);fps = 0;},1000)