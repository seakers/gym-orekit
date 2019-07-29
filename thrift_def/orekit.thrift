/**
 * The first thing to know about are types. The available types in Thrift are:
 *
 *  bool        Boolean, one byte
 *  i8 (byte)   Signed 8-bit integer
 *  i16         Signed 16-bit integer
 *  i32         Signed 32-bit integer
 *  i64         Signed 64-bit integer
 *  double      64-bit floating point value
 *  string      String
 *  binary      Blob (byte array)
 *  map<t1,t2>  Map from one type to another
 *  list<t1>    Ordered list of one type
 *  set<t1>     Set of unique elements of one type
 *
 * Did you also notice that Thrift supports C style comments?
 */

namespace java seakers.formationsimulator.thrift
namespace py gym_orekit.thrift


enum Frame {
  INERTIAL = 1,
  LVLH = 2
}

struct Vector3D {
  1: double x,
  2: double y,
  3: double z
}

typedef Vector3D DeltaV

typedef Vector3D Position

typedef Vector3D Velocity

struct SpacecraftState {
    1: Position position,
    2: Velocity velocity
}

struct GroundPosition {
    1: double latitude,
    2: double longitude
}

service Orekit {

    void reset(),

    void step(),

    bool done(),

    list<SpacecraftState> currentStates(),

    double getReward(),

    GroundPosition groundPosition(),

    list<GroundPosition> getFOV(),

    void sendLowLevelCommands(1: list<DeltaV> commandList),

    void sendHighLevelCommand(1: i32 command)

   // i32 calculate(1:i32 logid, 2:Work w) throws (1:InvalidOperation ouch),

   // oneway void zip()

}
