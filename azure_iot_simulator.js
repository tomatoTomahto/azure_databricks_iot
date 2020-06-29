/*
* IoT Hub Raspberry Pi NodeJS - Microsoft Sample Code - Copyright (c) 2017 - Licensed MIT
*/
const wpi = require('wiring-pi');
const Client = require('azure-iot-device').Client;
const Message = require('azure-iot-device').Message;
const Protocol = require('azure-iot-device-mqtt').Mqtt;
const BME280 = require('bme280-sensor');

const BME280_OPTION = {
  i2cBusNo: 1, // defaults to 1
  i2cAddress: BME280.BME280_DEFAULT_I2C_ADDRESS() // defaults to 0x77
};

const connectionString = 'HostName=<hub_hostname>;DeviceId=<hub_deviceid;SharedAccessKey=<access_key>';
const LEDPin = 4;
const directions = ["NW","W","SW"]; //,"SW","S","SE","E","NE"];
const angles = [-1,0,1]
var dir_idx = 0;

var sendingMessage = false;
var messageId = 0;
var client, sensor;
var blinkLEDTimeout = null;
var date = new Date();

function getWeatherMessage(cb) {
  dir_idx = (dir_idx + Math.round(Math.random()))%directions.length
  sensor.readSensorData()
    .then(function (data) {
      cb(JSON.stringify({
        timestamp: new Date().toISOString().replace(/T/, ' ').replace(/\..+/, ''),
        deviceId: 'WeatherCapture',
        temperature: data.temperature_C,
        humidity: data.humidity,
        windspeed: data.humidity/10,
        winddirection: directions[dir_idx]
      }), data.temperature_C > 30);
    })
    .catch(function (err) {
      console.error('Failed to read out sensor data: ' + err);
    });
}

function getEquipmentMessage(cb) {
  messageId++
  sensor.readSensorData()
    .then(function (data) {
      cb(JSON.stringify({
        timestamp: new Date().toISOString().replace(/T/, ' ').replace(/\..+/, ''),
        deviceId: 'WindTurbine-'  + messageId % 10,
        rpm: 60 * data.humidity/10 * 6 / 315,
        angle: angles[dir_idx] + data.humidity/10

      }), data.temperature_C > 30);
    })
    .catch(function (err) {
      console.error('Failed to read out sensor data: ' + err);
    });
}


function sendWeatherMessage() {
  if (!sendingMessage) { return; }

  getWeatherMessage(function (content, weatherAlert) {
    var message = new Message(content);
    message.properties.add('weatherAlert', weatherAlert.toString());
    console.log('Sending message: ' + content);
    client.sendEvent(message, function (err) {
      if (err) {
        console.error('Failed to send message to Azure IoT Hub');
      } else {
        blinkLED();
        console.log('Message sent to Azure IoT Hub');
      }
    });
  });
}

function sendEquipmentMessage() {
  if (!sendingMessage) { return; }

  getEquipmentMessage(function (content, equipmentAlert) {
    var message = new Message(content);
    message.properties.add('equipmentAlert', equipmentAlert.toString());
    console.log('Sending message: ' + content);
    client.sendEvent(message, function (err) {
      if (err) {
        console.error('Failed to send message to Azure IoT Hub');
      } else {
        blinkLED();
        console.log('Message sent to Azure IoT Hub');
      }
    });
  });
}

function onStart(request, response) {
  console.log('Try to invoke method start(' + request.payload + ')');
  sendingMessage = true;

  response.send(200, 'Successully start sending message to cloud', function (err) {
    if (err) {
      console.error('[IoT hub Client] Failed sending a method response:\n' + err.message);
    }
  });
}

function onStop(request, response) {
  console.log('Try to invoke method stop(' + request.payload + ')');
  sendingMessage = false;

  response.send(200, 'Successully stop sending message to cloud', function (err) {
    if (err) {
      console.error('[IoT hub Client] Failed sending a method response:\n' + err.message);
    }
  });
}

function receiveMessageCallback(msg) {
  blinkLED();
  var message = msg.getData().toString('utf-8');
  client.complete(msg, function () {
    console.log('Receive message: ' + message);
  });
}

function blinkLED() {
  // Light up LED for 500 ms
  if(blinkLEDTimeout) {
       clearTimeout(blinkLEDTimeout);
   }
  wpi.digitalWrite(LEDPin, 1);
  blinkLEDTimeout = setTimeout(function () {
    wpi.digitalWrite(LEDPin, 0);
  }, 500);
}

// set up wiring
wpi.setup('wpi');
wpi.pinMode(LEDPin, wpi.OUTPUT);
sensor = new BME280(BME280_OPTION);
sensor.init()
  .then(function () {
    sendingMessage = true;
  })
  .catch(function (err) {
    console.error(err.message || err);
  });

// create a client
client = Client.fromConnectionString(connectionString, Protocol);

client.open(function (err) {
  if (err) {
    console.error('[IoT hub Client] Connect error: ' + err.message);
    return;
  }

  // set C2D and device method callback
  client.onDeviceMethod('start', onStart);
  client.onDeviceMethod('stop', onStop);
  client.on('message', receiveMessageCallback);
  setInterval(sendWeatherMessage, 3000);
  setInterval(sendEquipmentMessage, 2000);
});
