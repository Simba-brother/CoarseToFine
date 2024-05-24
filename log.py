#! /usr/bin/env python

import logging,os
import sys

class Logger:
    def __init__(self, path, clevel = logging.DEBUG, Flevel = logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        #设置文件日志
        fh = logging.FileHandler(path,mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)

    def war(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def cri(self,message):
        self.logger.critical(message)

if __name__ =='__main__':
    logmml = Logger(path='./Log/mml.log', clevel=logging.ERROR, Flevel=logging.DEBUG)
    # data = 10/0
    logmml.debug('一个debug信息')
    logmml.info('一个info信息')
    logmml.war('一个warning信息')
    logmml.error('一个error信息')
    logmml.cri('一个致命critical信息')


