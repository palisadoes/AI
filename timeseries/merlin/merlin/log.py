#!/usr/bin/python3
"""Nagios check general library."""

import sys
import os
import getpass
import logging


__author__ = 'Peter Harrison (Colovore LLC.) <peter@colovore.com>'
__version__ = '0.0.1'


class Log(object):

    """Class has methods to process logging.

    Args:
        None

    Returns:
        None

    Functions:
        get_errors:
        get_packets:
        get_current_state:
        get_previous_state:
        state_filename:
        create_state_file:
        process_states:
        get_max_errors:
        get_perfdata:
        get_service_data:
        get_long_output:
        process_alert:
    """

    def __init__(self, is_error=True, app_name='', verbose=False):
        """Function for intializing the class.

        Args:
            is_error: Is this an error or not?
            app_name: Name of the app that is logging
            verbose: Verbose output if True

        Returns:
            None

        """
        # Initialize key variables
        self.is_error = is_error
        self.app_name = app_name
        self.verbose = verbose
        self.username = getpass.getuser()

        # create formatter and add it to the handlers
        self.formatter = logging.Formatter('%(asctime)s - %(name)s '
                                           '- %(levelname)s - %(message)s')

    def to_system(self, error_num, error_string, log_file=None):
        """Log to file and STDOUT. Hard exit if self.is_error is True.

        Args:
            error_num: Error number
            error_string: Descriptive error string
            log_file: File to log to

        Returns:
            None
        """
        # Log to file and stdout
        self.to_file(error_num, error_string, log_file=log_file)
        self.to_stdout(error_num, error_string)

        # End if Error
        if self.is_error:
            # All done
            sys.exit(2)

    def to_file(self, error_num, error_string, log_file=None):
        """Log to file.

        Args:
            error_num: Error number
            error_string: Descriptive error string
            log_file: File to log to

        Returns:
            None
        """
        # Log to file only if filename is given
        if log_file is None:
            return

        # If logfile doesn't exist then die
        if os.path.isfile(log_file) is False:
            error_string = (
                'Log file "%s" not found. Please fix. '
                'Other errors include: "%s"') % (log_file, error_string)
            self.to_stdout(error_num, error_string)
            sys.exit(2)

        # create logger
        logger_file = logging.getLogger(('%s_file') % (self.app_name))

        # Set logging levels to file and stdout
        logger_file.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        file_handle = logging.FileHandler(log_file)
        file_handle.setLevel(logging.DEBUG)

        # Add formatter to the handler
        file_handle.setFormatter(self.formatter)

        # add the handlers to the logger
        logger_file.addHandler(file_handle)

        # Get log message
        log_message = format_output(
            username=self.username,
            error_num=error_num,
            error_string=error_string,
            is_error=self.is_error)

        # Log to file, remove handler, close handler
        logger_file.debug(log_message)
        logger_file.removeHandler(file_handle)
        file_handle.close()

    def to_stdout(self, error_num, error_string):
        """Log to STDOUT.

        Args:
            error_num: Error number
            error_string: Descriptive error string

        Returns:
            None
        """
        # Print to STDOUT only if there is:
        # 1) An error or
        # 2) No errror and self.verbose is True
        if self.is_error is False:
            if self.verbose is False:
                return

        # create logger
        logger_stdout = logging.getLogger(('%s_console') % (self.app_name))

        # Set logging levels to file and stdout
        logger_stdout.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        stdout_handle = logging.StreamHandler()
        stdout_handle.setLevel(logging.DEBUG)

        # Add formatter to the handler
        stdout_handle.setFormatter(self.formatter)

        # add the handlers to the logger
        logger_stdout.addHandler(stdout_handle)

        # Get log message
        log_message = format_output(
            username=self.username,
            error_num=error_num,
            error_string=error_string,
            is_error=self.is_error)

        # Log to STDOUT, remove handler, close handler
        logger_stdout.debug('%s', log_message)
        logger_stdout.removeHandler(stdout_handle)
        stdout_handle.close()


def format_output(
        username=None, error_num=None, error_string=None, is_error=True):
    """Format message string.

    Args:
        error_num: Error number
        error_string: Descriptive error string
        is_error: Is this an error or not?

    Returns:
        log_message: Message to log
    """
    # Log the message
    if is_error:
        log_message = (
            'ERROR [%s] (%sE): %s') % (
                username, error_num, error_string)

    else:
        log_message = (
            'STATUS [%s] (%sS): %s') % (
                username, error_num, error_string)

    # Return
    return log_message


def log2die(error_num, error_string, is_error=True, verbose=False):
    """Log to STDOUT and file.

    Args:
        error_num: Error number
        error_string: Descriptive error string
        verbose: Log only if verbose is True

    Returns:
        None
    """
    # Initialize key variables
    app_name = 'fxcm'
    log_file = '/tmp/fx.log'

    # Create log file if it doesn't exist
    if os.path.isfile(log_file) is False:
        directory = os.path.dirname(log_file)
        os.makedirs(directory)
        with open(log_file, 'w'):
            os.chmod(log_file, 0o664)
            os.chmod(directory, 0o775)

    # Create the log object
    log_object = Log(is_error=is_error, app_name=app_name, verbose=verbose)

    # Log the message
    log_object.to_system(error_num, error_string, log_file=log_file)


def log2screen(error_num, error_string, verbose=True):
    """Log to STDOUT.

    Args:
        error_num: Error number
        error_string: Descriptive error string

    Returns:
        None
    """
    # Log to screen and file, but don't die
    log2die(error_num, error_string, is_error=False, verbose=verbose)
