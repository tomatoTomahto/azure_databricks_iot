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

const connectionString = 'HostName=sguptaiothub.azure-devices.net;DeviceId=sensors;SharedAccessKey=HyH5m3R4MmWE8KPIMcjDj7dkXtRpD79vVwYuKgMuito=';
const LEDPin = 4;

var sendingMessage = false;
var messageId = 0;
var client, sensor;
var blinkLEDTimeout = null;
var date = new Date();

function getSensorMessage(cb) {
  sensor.readSensorData()
    .then(function (data) {
      messageId++
      cb(JSON.stringify({
        facilityId: 'FAC-'  + messageId % 10,
        timestamp: new Date().toISOString().replace(/T/, ' ').replace(/\..+/, ''),
        temperature: data.temperature_C,
        humidity: data.humidity,
        pressure: data.humidity/2,
        moisture: data.humidity/1.5 + 5,
        oxygen: data.temperature_C / 2 + 15,
        radiation: data.temperature_C / 1.5 + data.humidity*1.3 + 12,
        conductivity: data.temperature_C / 1.2 + data.humidity*1.5 + 6
      }), data.temperature_C > 30);
    })
    .catch(function (err) {
      console.error('Failed to read out sensor data: ' + err);
    });
}

function sendSensorMessage() {
  if (!sendingMessage) { return; }

  getSensorMessage(function (content, weatherAlert) {
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
  setInterval(sendSensorMessage, 3000);
});
