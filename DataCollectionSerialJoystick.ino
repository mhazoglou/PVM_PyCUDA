#include <ax12.h>
#include <BioloidController.h>

/* 
   AX-Turret Code For Joystick
   Connect the joystick on pins analog pins 0 and 1 of the turret.
   
   -DEADBANDHIGH and DEADBANDLOW can be adjusted to the deadband of your particular joystick if needed
   -*_LIMIT defines are hard limits for pan and tilt to avoid the turret from damagint/over torque-ing itself
   -int speed can be varied to change the overall speed of the system
   
  ArbotiX Firmware - Commander Extended Instruction Set Example
  Copyright (c) 2008-2010 Vanadium Labs LLC.  All right reserved.
 
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the Vanadium Labs LLC nor the names of its 
        contributors may be used to endorse or promote products derived 
        from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL VANADIUM LABS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  Modifed Arbotix joystick code to write serial values of position upon
  receiving any single char over serial.
*/ 
 
 
//define pan and tilt servo IDs
#define PAN    1
#define TILT   2

// the F2 'C' bracket attached to the tilt servo creates a physical limitation to how far
// we can move the tilt servo. This software limit will ensure that we don't jam the bracket into the servo.
#define TILT_MAX 768 
#define TILT_MIN 256

//Upper/Lower limits for the pan servo - by defualt they are the normal 0-1023 (0-300) positions for the servo
#define PAN_MAX 1023 
#define PAN_MIN 0

//define analog pins that will be connected to the joystick pins
#define JOYPAN 0
#define JOYTILT 1

//Default/Home position. These positions are used for both the startup position as well as the 
//position the servos will go to when they lose contact with the commander
#define DEFAULT_PAN 512
#define DEFAULT_TILT 512

//generic deadband limits - not all joystics will center at 512, so these limits remove 'drift' from joysticks that are off-center.
#define DEADBANDLOW 480
#define DEADBANDHIGH 540

#define TIME_MSG_LEN  11   // time sync to PC is HEADER and unix time_t as ten ascii digits
#define TIME_HEADER  255   // Header tag for serial time sync message

//Include necessary Libraries to drive the DYNAMIXEL servos  
#include <ax12.h>
#include <BioloidController.h>
#include <stdlib.h>

int joyPanVal = 0;//current value of the pan joystick (analog 0) 0-1023
int joyTiltVal = 0;//current value of the tilt joystick (analog 1) 0-1023
int joyTiltMapped = 0;//tilt joystick value, mapped  to -speed - speed
int joyPanMapped = 0;//pan joystick value, mapped to -speed - speed

int speed = 30;//increase this to increase the speed of the movement

/* Hardware Construct */
BioloidController bioloid = BioloidController(1000000);  //create a bioloid object at a baud of 1MBps

int pan;    //set point position of the pan servo
int tilt;   //set point position of the tilt servo
int cur_pan;
int cur_tilt; 

const int MaxChars = 4; 
// number of character transfered serially for a single motion
char panStrValue[MaxChars+1];
char tiltStrValue[MaxChars+1];
int idx1 = 0;
int idx2 = 0;
boolean pan_serial = true;
char buffer [4];


void setup()
{
  Serial.begin(115200);    // open serial port
  pinMode(0,OUTPUT);     // setup user LED
  digitalWrite(0, HIGH); // turn user LED on to show the program is running
   
  
  // setup interpolation, slowly raise turret to a 'home' positon. 512 are the 'center' positions for both servos
  pan = DEFAULT_PAN;//load default pan value for startup
  tilt = DEFAULT_TILT;//load default tilt value for startup
  
  delay(1000);  //wait for the bioloid controller to intialize
  
  bioloid.poseSize = 2;            //2 servos, so the pose size will be 2
  bioloid.readPose();              //find where the servos are currently
  bioloid.setNextPose(PAN, DEFAULT_PAN);    //prepare the PAN servo to the default position
  bioloid.setNextPose(TILT, DEFAULT_TILT);  //preprare the tilt servo to the default position
  bioloid.interpolateSetup(2000);  //setup for interpolation from the current position to the positions set in setNextPose, over 2000ms
  while(bioloid.interpolating > 0) //until we have reached the positions set in setNextPose, execute the instructions in this loop
  {
    bioloid.interpolateStep();//move servos 1 'step
    delay(3);
  }
  
}


 
void loop()
{
  /*
  //read analog values from joysticks. Each variable will hold a value in between 0-1023 corresponding to the location of that axis of the joystick
  joyPanVal = analogRead(JOYPAN);
  joyTiltVal = analogRead(JOYTILT);
     
 
  //deadzone for pan jotystick - only change the pan value if the joystick value is outside the deadband
  if(joyPanVal > DEADBANDHIGH || joyPanVal < DEADBANDLOW)
  {
    //joyPanVal will hold a value between 0 and 1023 that correspods to the location of the joystick. The map() function will convert this value
    //into a value between speed and -speed. This value can then be added to the current panValue to incrementley move ths servo 
    joyPanMapped = map(joyPanVal, 0, 1023, -speed, speed);
    pan = pan + joyPanMapped;

    //enforce upper/lower limits for pan servo. This will ensure we do not move the servo to a position out of its physical bounds. 
    pan = max(pan, PAN_MIN);  //use the max() function to make sure the value never falls below PAN_MIN 
    pan = min(pan, PAN_MAX);  //use the min() function to make sute the value never goes above PAN_MAX 


  }
  
  //deadzone for tilt jotystick - only change the pan value if the joystick value is outside the deadband  
  if(joyTiltVal > DEADBANDHIGH || joyTiltVal < DEADBANDLOW)
  {
    //joyTiltVal will hold a value between 0 and 1023 that correspods to the location of the joystick. The map() function will convert this value
    //into a value between speed and -speed. This value can then be added to the current panValue to incrementley move ths servo 
    joyTiltMapped = map(joyTiltVal, 0, 1023, -speed, speed);
    tilt = tilt + joyTiltMapped;
         
    //enforce upper/lower limits for pan servo. This will ensure we do not move the servo to a position out of its physical bounds. 
    tilt = max(tilt, TILT_MIN);  //use the max() function to make sure the value never falls below TILT_MIN 
    tilt = min(tilt, TILT_MAX);  //use the min() function to make sute the value never goes above TILT_MAX 
   
 }
        
  //send pan and tilt goal positions to the pan/tilt servos 
  SetPosition(PAN,pan);
  SetPosition(TILT,tilt);

  delay(3); //delay to allow the analog-to-digital converter to settle before the next reading
  */
}

void serialEvent()
{
  while(Serial.available())
  {
    //bioloid.readPose();
    char ch = Serial.read();
    /*
    cur_pan = bioloid.getCurPose(PAN);
    cur_tilt = bioloid.getCurPose(TILT);
    Serial.print(itoa(cur_pan, buffer, 10));
    Serial.print(", ");
    Serial.println(itoa(cur_tilt, buffer, 10));
    //Serial.write(ch);
    */
    
    if(pan_serial){
      if(idx1 < MaxChars && isdigit(ch)) {
        panStrValue[idx1++] = ch;
      } 
      else{
        panStrValue[idx1] = 0;
        pan = atoi(panStrValue);
        pan = max(pan, PAN_MIN);  //use the max() function to make sure the value never falls below PAN_MIN 
        pan = min(pan, PAN_MAX);  //use the min() function to make sute the value never goes above PAN_MAX 
        SetPosition(PAN, pan);
        idx1 = 0;
        pan_serial = false;
        
      }
    } else {
      bioloid.readPose();
      if(idx2 < MaxChars && isdigit(ch)) {
        tiltStrValue[idx2++] = ch;
      }
      else {
        tiltStrValue[idx2] = 0;
        tilt = atoi(tiltStrValue);
        
        tilt = max(tilt, TILT_MIN);  //use the max() function to make sure the value never falls below TILT_MIN 
        tilt = min(tilt, TILT_MAX);  //use the min() function to make sute the value never goes above TILT_MAX //send pan and tilt goal positions to the pan/tilt servos
        SetPosition(TILT, tilt);
        idx2 = 0;
        pan_serial = true;
        
        cur_pan = bioloid.getCurPose(PAN);
        cur_tilt = bioloid.getCurPose(TILT);
        Serial.print(itoa(cur_pan, buffer, 10));
        Serial.print(", ");
        Serial.println(itoa(cur_tilt, buffer, 10));
      }
    }
    //delay(10); //delay to allow the analog-to-digital converter to settle before the next reading
    /*
    cur_pan = bioloid.getCurPose(PAN);
    cur_tilt = bioloid.getCurPose(TILT);
    Serial.print(itoa(cur_pan, buffer, 10));
    Serial.print(", ");
    Serial.println(itoa(cur_tilt, buffer, 10));
    */
  }
  
}
