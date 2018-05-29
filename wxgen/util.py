from __future__ import division
import numpy as np
import calendar
import datetime
import sys
import copy


DEBUG = False
COLORS = {"red": 31, "yellow": 33, "green": 32}


def random_weighted(weights, policy):
   """
   Randomly selects an index into an array based on weights

   Arguments:
      weights (np.array): An array of weights
      policy (str): A randomization policy
         'random': Randomly pick based using the weights as probabilities
         'top<N>': e.g. top3, pick a random (unweighted) value from the top N
   """
   if policy == "random":
      acc = np.cumsum(weights) / np.sum(weights)
      r = np.random.rand(1)
      temp = np.where(acc < r)[0]
      if len(temp) == 0:
         I = 0
      else:
         I = temp[-1]
   elif len(policy) > 3 and policy[0:3] == "top" and is_number(policy[3:]):
      N = int(policy[3:])
      if N >= len(weights):
         return np.random.randint(len(weights))
      else:
         """ Expand the search if there are ties in the top scores """
         Isorted = np.argsort(weights)[::-1]
         min_weight = weights[Isorted[N-1]]
         Nsearch = np.sum(weights >= min_weight)
         II = np.random.randint(Nsearch)
         I = Isorted[II]
   else:
      error("Invalid randomization policy '%s'" % policy)
   return I


def error(message):
   print("\033[1;31mError: " + message + "\033[0m")
   sys.exit(1)


def warning(message):
   print("\033[1;33mWarning: " + message + "\033[0m")


def debug(message, color="green"):
   col = COLORS[color]
   if DEBUG:
      print("\033[1;%imDebug: " % (col) + message + "\033[0m")


def parse_colors(color_string):
   """
   Turns a comma-separated string of colors into a list. E.g.
   "[0.6, 1, 0.3],k, red" turns into:
   [[0.6, 1, 0.3], 'k', 'red']
   """
   firstList = color_string.split(',')
   numList = []
   colors = []

   for string in firstList:
      if "[" in string:   # for rgba args
         if not numList:
            string = string.replace("[", "")
            numList.append(float(string))
         else:
            error("Invalid rgba arg \"{}\"".format(string))

      elif "]" in string:
         if numList:
            string = string.replace("]", "")
            numList.append(float(string))
            colors.append(numList)
            numList = []
         else:
            error("Invalid rgba arg \"{}\"".format(string))

      # append to rgba lists if present, otherwise grayscale intensity
      elif is_number(string):
         if numList:
            numList.append(float(string))
         else:
            colors.append(string)

      else:
         if not numList:  # string args and hexcodes
            colors.append(string)
         else:
            error("Cannot read color args.")
   return colors


def parse_dates(dates):
   return [int(date) for date in parse_numbers(dates, True)]


def parse_ints(ints):
   return [int(num) for num in parse_numbers(ints, True)]


def parse_numbers(numbers, isDate=False):
   """
   Convert a string into an array of numbers. allowable formats:
      num
      num1,num2,num3
      start:end
      start:step:end
   """
   if numbers is None:
      return None

   # Check if valid string
   if(any(char not in set('-01234567890.:,') for char in numbers)):
      error("Could not translate '" + numbers + "' into numbers")

   values = list()
   commaLists = numbers.split(',')
   for commaList in commaLists:
      colonList = commaList.split(':')
      if(len(colonList) == 1):
         value = float(colonList[0])
         if int(value) == value:
            values.append(int(value))
         else:
            values.append(value)
      elif(len(colonList) <= 3):
         start = float(colonList[0])
         step = 1
         if(len(colonList) == 3):
            step = float(colonList[1])
         end = float(colonList[-1])
         if(isDate):
            date = min(start, end)
            curr = list()
            while date <= max(start, end):
               curr.append(date)
               date = get_date(date, step)
            values = values + list(curr)
         else:
            if int(start) == start and int(end) == end and int(step) == step:
               # Generate integer list
               values = values + list(range(int(start), int(end)+1, int(step)))
            else:
               # arange does not include the end point:
               stepSign = step / abs(step)
               values = values + list(np.arange(start, end + stepSign*0.0001, step))
      else:
         error("Could not translate '" + numbers + "' into numbers")
      if(isDate):
         for i in range(0, len(values)):
            values[i] = int(values[i])
   return values


def resize(vec, size):
   """
   Resizes a vector such that it has the right size. This is done by repeating the vector
   in each dimension until the required size is reached. Note an error is thrown if 'size'
   is not a multiple of the size of vec.

   vec      A 1D or 2D numpy array
   size     A list of dimension sizes (e.g. [2,3])
   """
   if not isinstance(vec, (np.ndarray)):
      vec_resized = vec * np.ones(size)
   elif vec.shape[0] == size[0] and len(vec.shape) == 1:
      vec_resized = np.reshape(np.repeat(vec, size[1]), size)
   elif vec.shape[0] == 1 and len(vec.shape) == 1:
      vec_resized = vec*np.ones(size)
   else:
      # Check that the output dims are multiples of input dims
      assert(size[0] % vec.shape[0] == 0)
      assert(size[1] % vec.shape[1] == 0)
      vec_resized = np.tile(vec, (size[0] // vec.shape[0], size[1] // vec.shape[1]))
   return vec_resized


def date_to_unixtime(date):
   year = date // 10000
   month = date // 100 % 100
   day = date % 100
   ut = calendar.timegm(datetime.datetime(year, month, day).timetuple())
   return ut


def unixtime_to_date(unixtime):
   dt = datetime.datetime.utcfromtimestamp(int(unixtime))
   date = dt.year * 10000 + dt.month * 100 + dt.day
   return date


def correlation(ar1, ar2, axis):
   """
   Computes the correlation between two multi-dimensional arrays
   """
   ar1mean = np.resize(np.mean(ar1, axis=axis), ar1.shape)
   ar2mean = np.resize(np.mean(ar2, axis=axis), ar2.shape)

   cov = np.mean((ar1 - ar1mean) * (ar2 - ar2mean), axis=axis)
   var1 = np.var(ar1, axis=axis)
   var2 = np.var(ar2, axis=axis)

   return cov / np.sqrt(var1) / np.sqrt(var2)


def get_date(date, diff):
   """ Date calculation: Adds 'diff' to 'date'

   Arguments:
      date (int): An integer of the form YYYYMMDD
      diff (int): Number of days to add to date

   Returns:
      int: A new date in the form YYYYMMDD
   """
   year = int(date // 10000)
   month = int(date // 100 % 100)
   day = int(date % 100)
   date2 = datetime.datetime(year, month, day, 0) + datetime.timedelta(diff)
   return int(date2.strftime('%Y%m%d'))


def day_of_year(unixtime):
   """
   Arguments:
      unixtime (int): Number of seconds since 1970-01-01

   Returns:
      int: Day of year
   """
   day = int(datetime.datetime.fromtimestamp(unixtime).strftime('%j'))
   return day


def get_i_j(lats, lons, lat, lon):
   """
   Finds the nearest neighbour in a lat lon grid. If the point is outside the grid, the nearest
   point within the grid is still returned.

   Arguments:
      lats (np.array): 2D array of latitudes
      lons (np.array): 2D array of longitude
      lat (float): Loopup latitude
      lon (float): Loopup longitude

   Returns:
      I (int): First index into lats/lons arrays
      J (int): Second index into lats/lons arrays
   """
   dist = distance(lat, lon, lats, lons)
   indices = np.unravel_index(dist.argmin(), dist.shape)
   X = lats.shape[0]
   Y = lats.shape[1]
   I = indices[0]
   J = indices[1]
   if(indices[0] == 0 or indices[0] >= X-1 or indices[1] == 0 or indices[1] >= Y-1):
      debug("Lat/lon %g,%g outside grid" % (lat, lon))
   return I, J


def distance(lat1, lon1, lat2, lon2):
   """
   Computes the great circle distance between two points using the
   haversine formula. Values can be vectors.
   """
   # Convert from degrees to radians
   pi = 3.14159265
   lon1 = lon1 * 2 * pi / 360
   lat1 = lat1 * 2 * pi / 360
   lon2 = lon2 * 2 * pi / 360
   lat2 = lat2 * 2 * pi / 360
   dlon = lon2 - lon1
   dlat = lat2 - lat1
   a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
   c = 2 * np.arcsin(np.sqrt(a))
   distance = 6.367e6 * c
   return distance


def is_number(s):
   """ Returns true if x is a scalar number """
   try:
      float(s)
      return True
   except ValueError:
      return False


def climatology(array, window=1, use_future_years=False):
   """
   Arguments:
      array (np.array): 2D array with dimensions leadtime, member
      window (int): Use an average across this many days
      use_future_years (bool): Use day 366 in the calculation for day 1, etc

   Returns:
      np.array: 1D array with climatology for each leadtime
   """
   import astropy.convolution
   if window < 1:
      error("Cannot have a window size of less than 1")
   elif window == 1:
      clim = np.nanmean(array, axis=1)
   else:
      # extend: Use the first and last values as padding outside the array
      clim = astropy.convolution.convolve(np.nanmean(array, axis=1), 1.0/window*np.ones(window), "extend")

   if use_future_years and clim.shape[0] > 365:
      temp = copy.deepcopy(clim)
      for i in range(temp.shape[0]):
         I = range(i % 365, temp.shape[0], 365)
         clim[i] = np.mean(temp[I])

   return clim


def nanpercentile(data, pers):
   I = np.where(np.isnan(data.flatten()) == 0)[0]
   p = np.nan
   if(len(I) > 0):
      p = np.percentile(data.flatten()[I], pers)
   return p


def normalize(array, window=11, normalize_variance=True):
   """
   Arguments:
      array (np.array): 2D array (time, member)
      window (int): Window length to compute climatology
      normalize_variance (bool): Adjust variance

   Returns:
      np.array: Array of normalized values (same size as input array)
   """
   N = array.shape[1]

   """
   Remove climatology so we can look at annomalies. Use separate obs and fcst climatology
   otherwise the fcst variance is higher because obs gets the advantage of using its own
   climatology.
   """
   clim = climatology(array, window, use_future_years=True)
   values = copy.deepcopy(array)
   for i in range(0, N):
      values[:, i] = (values[:, i] - clim)

   if normalize_variance and array.shape[1] > 2:
      """
      This removes any seasonally varying variance, which can cause the 1-year variance to be
      larger than the 1/2 year variance, because the 1/2 year variance samples the summer months
      more often than the winter months, because of the windowing approach. Also, this
      normalization does not guarantee that the std of the whole timeseries is 1, therefore in
      the plot, don't expect the first point to be 1.

      The timeseries is scaled up again to match the average anomaly variance in the timeseries.
      """
      std = np.nanstd(array, axis=1)
      if np.min(std) == 0:
         warning("Standard deviation of 0 at one or more days. Not normalizing variance")
      else:
         meanstd = np.nanmean(std)
         for i in range(0, N):
            values[:, i] = values[:, i] / std * meanstd

   return values


def clean(variable, dtype=None):
   """
   Changes masked values to np.nan. Converts to float if input is integer, since
   integers cannot represent nan values.

   Arguments:
      variable: NetCDF4 variable
      dtype: output type

   Return:
      np.array: masked values are converted to np.nan
   """
   if dtype is None:
      if variable.dtype in ["int", "int32", "int64"]:
         values = np.ma.filled(variable[:].astype(float), np.nan)
      else:
         values = np.ma.filled(variable[:], np.nan)
   else:
      if variable.dtype == dtype:
         values = np.ma.filled(variable[:], np.nan)
      else:
         values = np.ma.filled(variable[:].astype(dtype), np.nan)
   return values


def nprange(data, axis):
   return np.max(data, axis=axis) - np.min(data, axis=axis)
