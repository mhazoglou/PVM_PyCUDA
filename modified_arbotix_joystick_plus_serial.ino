/* 
   Modifed Arbotix joystick code to handle serial values
   instead of joystick input.
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

//Default/Home position. These positions are used for both the startup position as well as the 
//position the servos will go to when they lose contact with the commander
#define DEFAULT_PAN 512
#define DEFAULT_TILT 512

//generic deadband limits - not all joystics will center at 512, so these limits remove 'drift' from joysticks that are off-center.
//#define DEADBANDLOW 480
//#define DEADBANDHIGH 540


//Include necessary Libraries to drive the DYNAMIXEL servos  
#include <ax12.h>
#include <BioloidController.h>

int speed = 50;//increase this to increase the speed of the movement

/* Hardware Construct */
BioloidController bioloid = BioloidController(1000000);  //create a bioloid object at a baud of 1MBps

int pan;    //current position of the pan servo
int tilt;   //current position of the tilt servo
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
  Serial.begin(9600);    // open serial port
 
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
//  SetPosition(PAN, pan);
//  SetPosition(TILT, tilt);
  
}


 
void loop()
{}

void serialEvent()
{
  while(Serial.available())
  {
    char ch = Serial.read();
    //Serial.write(ch);
    
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
        //SetPosition(TILT, tilt);
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
        //SetPosition(PAN, pan);
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
  }
  
}
