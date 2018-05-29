import sys
import argparse
import numpy as np
import wxgen.climate_model
import wxgen.database
import wxgen.downscaler
import wxgen.trajectory
import wxgen.generator
import wxgen.metric
import wxgen.output
import wxgen.plot
import wxgen.parameters
import wxgen.version


def main(argv):
   parser, sp = get_parsers()

   # Show help message when no options are provided
   if len(argv) < 2:
      parser.print_help()
      return
   elif len(argv) == 2 and argv[1] == "--version":
      print(wxgen.version.__version__)
      return
   elif len(argv) == 2 and argv[1] in sp.keys():
      sp[argv[1]].print_help()
      return

   args = parser.parse_args(argv[1:])

   """
   Run commands
   """
   wxgen.util.DEBUG = args.debug
   if args.command == "sim":
      db = get_db(args)

      V = len(db.variables)
      if args.initial is None:
         initial_state = None
      else:
         initial_state = np.array(wxgen.util.parse_numbers(args.initial))
         if len(initial_state) != V:
            wxgen.util.error("Initial state must match the number of variables (%d)" % (V))

      # Check that the number of weights equals the number of variables in the database
      if args.weights is not None and len(args.weights) != V:
         wxgen.util.error("Number of weights (-w) must match number of variables (-v)")

      # Generate trajectories
      metric = get_metric(args, db)
      generator = wxgen.generator.LargeScale(db, metric)
      generator.prejoin = args.prejoin
      generator.policy = args.policy
      generator.stagger = args.stagger
      generator.start_date = args.init_date

      # Allowable dates from database
      generator.db_start_date = args.start_date
      generator.db_end_date = args.end_date

      Ntimesteps = int(np.ceil(args.t * 1.0 / db.timestep * 86400))
      trajectories = generator.get(args.n, Ntimesteps, initial_state)

      # Create output
      output = wxgen.output.Netcdf(args.filename)
      output.lat = args.lat
      output.lon = args.lon
      output.write_indices = args.write_indices
      output.write(trajectories, db, args.scale, start_date=args.init_date)

   elif args.command == "downscale":
      db = get_db(args)
      output = wxgen.output.Netcdf(args.filename)
      method = wxgen.downscaler.get(args.method)
      parameters = wxgen.parameters.Parameters(args.parameters)
      output.write_downscaled(db, parameters, method, args.var, args.member)

   elif args.command == "truth":
      db = get_db(args)
      if args.n is None and args.t is None:
        trajectories = [db.get_truth(args.start_date, args.end_date, args.which_leadtime)]
      else:
         """
         Create a number of ensemble members, by sampling long timeseries from database (no
         joining). This is determined by -sd, -ed, -n, and -t. -sd -ed sets which dates are allowed
         to use from the database. Then -n sets the number of scenarios.

         We only want to create scenarios that all start at the same time of the year.
         """
         if args.n is None:
            wxgen.util.error("-n not specified")
         if args.t is None:
            wxgen.util.error("-t not specified")

         # Determine allowable dates from database
         start_date = args.start_date
         end_date = args.end_date
         if args.start_date is None:
            start_date = wxgen.util.unixtime_to_date(np.min(db.inittimes))
         if args.end_date is None:
            end_date = wxgen.util.unixtime_to_date(np.max(db.inittimes))
         dates = np.array(wxgen.util.parse_dates("%d:%d" % (start_date, end_date)))

         # Figure out which time indices are possible starting dates, by finding dates that have the
         # same day of year as the first date of the allowable dates
         months = dates / 100 % 100
         days = dates % 100
         start_day = args.init_date % 100
         start_month = args.init_date / 100 % 100
         Ipossible_start_days = np.where((months == start_month) & (days == start_day))[0]
         if args.n > len(Ipossible_start_days):
            wxgen.util.warning("Not enough possible starting days (%d < %d)" % (len(Ipossible_start_days), args.n))

         trajectories = list()
         for n in range(min(args.n, len(Ipossible_start_days))):
            s = dates[Ipossible_start_days[n]]
            e = wxgen.util.get_date(s, args.t)
            if e < end_date:
               wxgen.util.debug("Member %d dates: %d - %d" % (n, s, e))
               trajectory = db.get_truth(s, e, args.which_leadtime)
               trajectories += [trajectory]
            else:
               wxgen.util.debug("Skipping member %d: Goes outside date range" % n, "yellow")
      if len(trajectories) == 0:
         wxgen.util.error("Could not create any trajectories that are long enough")

      output = wxgen.output.Netcdf(args.filename)
      output.lat = args.lat
      output.lon = args.lon
      output.write_indices = args.write_indices
      output.write(trajectories, db, args.scale, start_date=args.init_date)

   elif args.command == "verif":
      plot = wxgen.plot.get(args.metric)()
      plot.filename = args.filename
      plot.xlim = args.xlim
      plot.xlog = args.xlog
      plot.ylim = args.ylim
      plot.ylog = args.ylog
      plot.vars = args.vars
      plot.thresholds = args.thresholds
      plot.fig_size = args.fs
      plot.clim = args.clim
      plot.cmap = args.cmap
      plot.lat = args.lat
      plot.lon = args.lon
      plot.scale = args.scale
      plot.line_styles = args.styles
      plot.line_colors = args.colors
      plot.line_widths = args.widths
      plot.marker_face_colors = args.mfc
      plot.markers = args.markers
      plot.marker_sizes = args.ms
      plot.grid = not args.nogrid
      plot.dpi = args.dpi

      if args.timescale is not None:
         if not plot.supports_timescale:
            wxgen.util.error("Plot does not support -ts")
         plot.timescale = args.timescale

      if args.timemod is not None:
         if not plot.supports_timemod:
            wxgen.util.error("Plot does not support -tm")
         plot.timemod = args.timemod

      transform = get_transform(args)
      if transform is not None:
         if not plot.supports_transform:
            wxgen.util.error("Plot does not support -tr")
         plot.transform = transform

      if args.aggregator is not None:
         aggregator = get_aggregator(args.aggregator)
         if aggregator is not None:
            if not plot.supports_time_aggregator and not plot.supports_ens_aggregator:
               wxgen.util.error("Plot does not support -a")
            plot.time_aggregator = aggregator
            plot.ens_aggregator = aggregator
      else:
         if args.ens_aggregator is not None:
            if not plot.supports_ens_aggregator:
               wxgen.util.error("Plot does not support -ea")
            aggregator = get_aggregator(args.ens_aggregator)
            plot.ens_aggregator = aggregator
         if args.time_aggregator is not None:
            if not plot.supports_time_aggregator:
               wxgen.util.error("Plot does not support -ta")
            aggregator = get_aggregator(args.time_aggregator)
            plot.time_aggregator = aggregator

      sims = [wxgen.database.Netcdf(file, None) for file in args.files]

      # Check that we aren't trying to use a variable out of range
      sim_num_variables = [len(sim.variables) for sim in sims]
      if args.vars is not None and max(args.vars) >= max(sim_num_variables):
         wxgen.util.error("One or more inputs has only %d variables. Variable %d is outside range." % (max(sim_num_variables), max(args.vars)))

      if args.legend is not None:
         labels = [lab.replace('_', ' ') for lab in args.legend.split(',')]
         if len(labels) != len(sims):
            wxgen.util.error("Number of legend labels (%d) does not equal number of simulations (%d)" % (len(labels), len(sims)))
         for i in range(len(sims)):
            sims[i].label = labels[i]
      plot.plot(sims)


def get_parsers():
   """ Sets up the argparser object for wxgen

   Returns:
      parser: The main argparse object
      sp: A list of subparsers
   """
   parser = argparse.ArgumentParser(prog="wxgen", description="Hybrid weather generator, combining stochastic and physical modelling")
   parser.add_argument('--version', action="version", version=wxgen.version.__version__)
   subparsers = parser.add_subparsers(title="Choose one of these commands", dest="command")

   """
   Simulation driver
   """
   sp = dict()
   sp["sim"] = subparsers.add_parser('sim', help='Create simulated scenarios')
   sp["sim"].add_argument('-n', metavar="NUM", type=int, help="Number of trajectories", required=True)
   sp["sim"].add_argument('-t', metavar="DAYS", type=int, help="Length of trajectory", required=True)
   sp["sim"].add_argument('-m', default="rmsd", help="Metric for matching states", dest="metric", choices=get_module_names(wxgen.metric))
   sp["sim"].add_argument('-w', type=wxgen.util.parse_numbers, help="Weights for each variable when joining (comma-separated)", dest="weights")
   sp["sim"].add_argument('-i', help="Initial state (aggregated values for each variable). If unspecified, choose random starting states.", dest="initial")
   sp["sim"].add_argument('-j', type=int, metavar="NUM", help="How many times should segments be prejoined?", dest="prejoin")
   sp["sim"].add_argument('-b', type=int, metavar="DAYS", help="Length of database bins", dest="bin_width")
   sp["sim"].add_argument('-p', default="top5", metavar="POLICY", help="Randomization policy. One of 'random', 'top<N>'", dest="policy")
   sp["sim"].add_argument('-g', help="Randomly truncate the first segment so that potential jumps are staggered", dest="stagger", action="store_true")

   """
   Downscaler driver
   """
   sp["downscale"] = subparsers.add_parser('downscale', help='Create downscaled scenarios')
   sp["downscale"].add_argument('db', help="NetCDF file with large scale scenarios (i.e. output of wxgen sim)", nargs=1)
   sp["downscale"].add_argument('-m', default="qq", help="Downscaling method", dest="method", choices=get_module_names(wxgen.downscaler))
   sp["downscale"].add_argument('-o', metavar="FILENAME", help="Output filename", dest="filename", required=True)
   # Parameters are required (even for nearest neighbour) because they contain the grid definition
   sp["downscale"].add_argument('-p', help="Parameter file used by downscaling method", required=True, dest="parameters")
   sp["downscale"].add_argument('-e', type=int, help="Which ensemble member to output?", dest="member", required=True)
   # sp["downscale"].add_argument('-n', metavar="VARIABLE", help="Name of variable to downscale", required=True, dest="variable")
   sp["downscale"].add_argument('-v', type=int, metavar="INDEX", help="Which variable to downscale?  Use index, starting at 0.", required=True, dest="var")

   """
   Truth trajetory driver
   """
   sp["truth"] = subparsers.add_parser('truth', help='Create truth scenario')
   sp["truth"].add_argument('-n', metavar="NUM", type=int, help="Number of trajectories (if -n and -t are unspecified, create one trajectory with all data)")
   sp["truth"].add_argument('-t', metavar="DAYS", type=int, help="Length of trajectory")
   sp["truth"].add_argument('-ltd', metavar="DAY", default=0, type=int, help="Which lead time should be used as truth?", dest="which_leadtime")

   """
   Verification driver
   """
   sp["verif"] = subparsers.add_parser('verif', help='Verify trajectories')
   sp["verif"].add_argument('files', help="Input files", nargs="+")
   sp["verif"].add_argument('-fs', type=wxgen.util.parse_ints, default=[10, 5], help="Figure size: width,height")
   sp["verif"].add_argument('-m', help="Verification metric", dest="metric", required=True, choices=get_module_names(wxgen.plot))
   sp["verif"].add_argument('-o', metavar="FILENAME", help="Write output figure to this filename", dest="filename")
   sp["verif"].add_argument('-xlim', type=wxgen.util.parse_ints, help="x-axis limits: lower,upper (e.g. 0,10)")
   sp["verif"].add_argument('-xlog', help="Use a log scale for the X-axis?", action="store_true")
   sp["verif"].add_argument('-ylim', type=wxgen.util.parse_ints, help="y-axis limits: lower,upper (e.g. 0,10)")
   sp["verif"].add_argument('-ylog', help="Use a log scale for the Y-axis?", action="store_true")
   sp["verif"].add_argument('-r', dest="thresholds", help="Thresholds for use in some plots. Comma-separated.", required=False, type=wxgen.util.parse_numbers)
   sp["verif"].add_argument('-tr', dest="transform", help="Transform the variable", choices=get_module_names(wxgen.transform))
   aggregators = [mod for mod in get_module_names(wxgen.aggregator) if mod != "quantile"]
   sp["verif"].add_argument('-a', dest="aggregator", help="Aggregator for all dimensions (overrides -ta -ea). One of: " + ', '.join(aggregators) + " or a number between 0 and 1 representing a quantile")
   sp["verif"].add_argument('-ta', dest="time_aggregator", help="Aggregator for time dimension")
   sp["verif"].add_argument('-ea', dest="ens_aggregator", help="Aggregator for ensemble dimension")
   sp["verif"].add_argument('-clim', type=wxgen.util.parse_numbers, help="Colorbar limits (lower,upper)")
   sp["verif"].add_argument('-cmap', help="Colormap to use in some plots (e.g. jet, RdBu, Blues_r)")
   sp["verif"].add_argument('-tm', type=int, help="Time modulus (in days)", dest="timemod")
   sp["verif"].add_argument('-ts', type=int, help="Time scale (in days)", dest="timescale")
   sp["verif"].add_argument('-leg', help="Replace labels with these comma-separated names in the legend (same order as input files)", dest="legend")
   sp["verif"].add_argument('-ls', help="Line style for plots (comma-separated, e.g. -,o-,--)", dest="styles")
   sp["verif"].add_argument('-lc', type=wxgen.util.parse_colors, help="Line colors for plots (comma-separated, e.g.  red,[1,0.7,0.3],blue)", dest="colors")
   sp["verif"].add_argument('-lw', type=wxgen.util.parse_numbers, help="Line widths for plots (comma-separated, e.g.  2,2,1)", dest="widths")
   sp["verif"].add_argument('-mfc', type=wxgen.util.parse_colors, help="Marker face colors", dest="mfc")
   sp["verif"].add_argument('-marker', help="Markers (e.g. o,None,.)", dest="markers")
   sp["verif"].add_argument('-ms', type=wxgen.util.parse_numbers, help="Marker sizes (e.g. 1,1,3)", dest="ms")
   sp["verif"].add_argument('-nogrid', help="Turn grid in figure off", action="store_true", dest="nogrid")
   sp["verif"].add_argument('-dpi', default=200, type=int, help="Dots per inch in output image", dest="dpi")

   for driver in ["sim", "truth", "downscale"]:
      pass

   for driver in ["sim", "truth", "verif"]:
      sp[driver].add_argument('-s', default="large", help="Output scale", choices=["agg", "large", "small"], dest="scale")
      sp[driver].add_argument('-lat', type=float, help="Lookup latitude")
      sp[driver].add_argument('-lon', type=float, help="Lookup longitude")
      # Note, a different definition of -v for downscale
      sp[driver].add_argument('-v', metavar="INDICES", help="Which variables to use? Use indices, starting at 0.", required=False, type=wxgen.util.parse_ints, dest="vars")

   for driver in ["sim", "downscale"]:
       sp[driver].add_argument('-rs', type=int, help="Random number seed", dest="seed")

   for driver in ["sim", "truth"]:
      sp[driver].add_argument('-db', metavar="FILENAME", help="Filename of NetCDF database")
      sp[driver].add_argument('-dbtype', metavar="TYPE", default="netcdf", help="Database type (netcdf, random, lorenz63). Either specify database name with -db, or set -dbtype to one of 'random' or 'lorenz63' to test idealized weather scenarios.")
      sp[driver].add_argument('-sd', metavar="YYYYMMDD", type=int, help="Earliest date to use from database", dest="start_date")
      sp[driver].add_argument('-ed', metavar="YYYYMMDD", type=int, help="Latest date to use from database", dest="end_date")
      sp[driver].add_argument('-o', metavar="FILENAME", help="Filename to write output to", dest="filename", required=True)
      sp[driver].add_argument('-wl', type=int, default=0, metavar="NUM", help="Number of wavelet levels.  If 0 (default), don't use wavelets.", dest="wavelet_levels")
      sp[driver].add_argument('--write-indices', help="Write segment indicies into output. Used for debugging and analysis.", dest="write_indices", action="store_true")
      sp[driver].add_argument('-d', type=int, default=20170101, help="Start date of simulation (YYYYMMDD)", dest="init_date")

   for driver in sp.keys():
      sp[driver].add_argument('--debug', help="Display debug information", action="store_true")
      sp[driver].add_argument('-mem', type=float, help="Maximum memory allocation when reading from database (GB)")
   return parser, sp


def get_db(args):
   # Set up database
   if args.command == "sim" and args.seed is not None:
      np.random.seed(args.seed)

   model = get_climate_model(args)
   dbtype = "netcdf"
   if hasattr(args, "dbtype"):
      dbtype = args.dbtype

   if dbtype == "netcdf":
      if args.db is None:
         wxgen.util.error("Missing -db")
      if hasattr(args, "vars"):
         vars = args.vars
      else:
         vars = None
      if isinstance(args.db, list):
          assert(len(args.db) == 1)
          dbname = args.db[0]
      else:
          dbname = args.db
      db = wxgen.database.Netcdf(dbname, vars, model=model, mem=args.mem)
   else:
      # Don't use args.t as the segment length, because then you never get to join
      # Don't use args.n as the number of segments, because then you never get to join
      if dbtype is None or dbtype == "random":
         db = wxgen.database.Random(model=model)
      elif dbtype == "lorenz63":
         db = wxgen.database.Lorenz63(10, 50, model=wxgen.climate_model.Zero())
      else:
         wxgen.util.error("Cannot understand -dbtype %s" % dbtype)

   if hasattr(args, "wavelet_levels"):
      db.wavelet_levels = args.wavelet_levels

   if args.debug:
      db.info()

   return db


def get_metric(args, db):
   if args.weights is not None:
      weights = args.weights
      if args.wavelet_levels is not None:
         """
         If we are using wavelets, then the weights in the metric must be repeated such that each
         wavelet component gets the same weight for a given variable.
         """
         NX, NY = db.get_wavelet_size()
         N = int(NX * NY)
         weights = np.repeat(weights, N)

      if args.metric == "rmsd":
         metric = wxgen.metric.Rmsd(weights)
      elif args.metric == "max":
         metric = wxgen.metric.Max(weights)
      elif args.metric == "mad":
         metric = wxgen.metric.Mad(weights)
      elif args.metric == "exp":
         metric = wxgen.metric.Exp(weights)
   else:
      metric = wxgen.metric.get(args.metric)

   return metric


def get_climate_model(args):
   model = None
   try:
      # args might not have bin_width
      if args.bin_width is not None:
         model = wxgen.climate_model.Bin(args.bin_width)
   except:
      pass
   return model


def get_transform(args):
   transform = None
   if args.transform is not None:
      transform = wxgen.transform.get(args.transform)
   return transform


def get_aggregator(name):
   aggregator = None
   if name is not None:
      aggregator = wxgen.aggregator.get(name)
   return aggregator


def get_module_names(module):
   """
   Returns a list of strings, one for each class in the module
   """
   return [x[0].lower() for x in module.get_all() if "wxgen." + x[0].lower() != module.__name__]


if __name__ == '__main__':
   main()
