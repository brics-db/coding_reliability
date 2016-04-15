#include "timer.h"
#include "timestatistics.h"
#include <stdexcept>
#include <stdio.h>

TimeStatistics::TimeStatistics(Statistics* stats, TimeType type)
{
  if(stats==NULL){
    _statistics = new Statistics();
    _delete_statistics = true;
  }else{
    _delete_statistics = false;
    _statistics = stats;
  }
  _timetype = type;
}

TimeStatistics::~TimeStatistics()
{
  if(_delete_statistics)
    delete _statistics;
}

void
TimeStatistics::start(int index)
{
  switch(_timetype)
  {
    case CPU_CLOCK_TIME:
      startTimer_CPU(&_mtimes[index]);
      break;
    case CPU_WALL_TIME:
      startTimer_CPUWall(&_mtimes[index]);
      break;
    case GPU_TIME:
      startTimer_GPU(&_mtimes[index]);
      break;
    default:
      throw std::invalid_argument("Unhandled TIME TYPE.");
  }
  _started[index] = true;
}

void
TimeStatistics::stop(int index)
{
  if(_started[index]==false)
    throw std::runtime_error("Timer must be started before.");
  double eltime = stopTimer(&_mtimes[index]);
  _statistics->process( index, eltime );
  _started[index] = false;
}
/**
 * Add a time metric for which statistics are generated after timer has stopped.
 * @param label Metric label/short description. If already exists metric data will be resetted.
 * @param unit Metric unit.
 * @param invert Inverts time metric, so we have 1/time-scale.
 * @param factor Scales the time by this number.
 * @return Unique index of metric.
 */
int
TimeStatistics::add(const std::string& label, const std::string& unit, bool invert, double factor)
{
  int index = _statistics->add(label, unit, invert, factor);
  _mtimes.resize(_statistics->getLength());
  _started.resize(_statistics->getLength());
  _started[index] = false;
  return index;
}
/**
 * Add a time metric for which statistics are generated after timer has stopped.
 * @param label Metric label/short description. If already exists metric data will be appended.
 * @param unit Metric unit.
 * @param invert Inverts time metric, so we have 1/time-scale.
 * @param factor Scales the time by this number.
 * @return Unique index of metric.
 */
int
TimeStatistics::append(const std::string& label, const std::string& unit, bool invert, double factor)
{
  int index = _statistics->append(label, unit, invert, factor);
  _mtimes.resize(_statistics->getLength());
  _started.resize(_statistics->getLength());
  _started[index] = false;
  return index;
}
