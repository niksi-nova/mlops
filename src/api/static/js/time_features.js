/* Port of src/features/time_features.py's add_time_of_day_features(),
 * for the one case this site needs: deriving hour/minute_of_day/hour_sin/
 * hour_cos/is_peak_am/is_peak_pm from a minute-of-day the user picked.
 *
 * Python (the source of truth):
 *   seconds_of_day = seconds_col % 86400
 *   hour = seconds_of_day // 3600
 *   minute_of_day = seconds_of_day // 60
 *   hour_sin = sin(2*pi*hour/24)
 *   hour_cos = cos(2*pi*hour/24)
 *   is_peak_am = peak_am[0] <= hour <= peak_am[1]
 *   is_peak_pm = peak_pm[0] <= hour <= peak_pm[1]
 *
 * This site's slider picks minute_of_day directly (0-1439), so hour is
 * derived from it rather than the other way around - consistent because
 * hour = floor(minute_of_day / 60) exactly matches floor(seconds/3600)
 * when minute_of_day = floor(seconds/60), for integer inputs.
 */

const PEAK_AM = [8, 10];
const PEAK_PM = [17, 20];

function deriveTimeFeatures(minuteOfDay) {
  const hour = Math.floor(minuteOfDay / 60);
  const hourSin = Math.sin((2 * Math.PI * hour) / 24.0);
  const hourCos = Math.cos((2 * Math.PI * hour) / 24.0);
  const isPeakAm = hour >= PEAK_AM[0] && hour <= PEAK_AM[1] ? 1 : 0;
  const isPeakPm = hour >= PEAK_PM[0] && hour <= PEAK_PM[1] ? 1 : 0;
  return {
    hour,
    minute_of_day: minuteOfDay,
    hour_sin: hourSin,
    hour_cos: hourCos,
    is_peak_am: isPeakAm,
    is_peak_pm: isPeakPm,
  };
}

function formatMinuteOfDay(minuteOfDay) {
  const h = Math.floor(minuteOfDay / 60);
  const m = minuteOfDay % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
