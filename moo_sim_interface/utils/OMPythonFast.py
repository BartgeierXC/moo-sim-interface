import os
import platform
import subprocess

from OMPython import ModelicaSystem


def _find_mingw32_make() -> str | None:
    """Return the absolute path to mingw32-make.exe, trying OPENMODELICAHOME first."""
    if platform.system() == 'Windows':
        omhome = os.environ.get('OPENMODELICAHOME', '')
        if omhome:
            candidate = os.path.join(omhome, 'tools', 'msys', 'ucrt64', 'bin', 'mingw32-make.exe')
            if os.path.exists(candidate):
                return candidate
        # Fall back to PATH lookup
        import shutil
        return shutil.which('mingw32-make') or shutil.which('make')
    else:
        import shutil
        return shutil.which('make')


class ModelicaSystemFast(ModelicaSystem):
    def buildModel(self, variableFilter=None, verbose=True):
        """Build the model, falling back to a manual make call if OMC's internal build fails."""
        super().buildModel(variableFilter, verbose)

        # Check whether OMC's internal build succeeded
        if self.xmlFile and os.path.exists(self.xmlFile):
            return  # build succeeded – nothing to do

        # --- Fallback: run make manually ---
        build_dir = self.tempdir
        model_name = self.modelName
        exe_name = f'{model_name}.exe' if platform.system() == 'Windows' else model_name
        exe_path = os.path.join(build_dir, exe_name)
        makefile = os.path.join(build_dir, f'{model_name}.makefile')

        if not os.path.exists(makefile):
            print(f'[fallback build] Makefile not found: {makefile}')
            return

        make_cmd = _find_mingw32_make()
        if make_cmd is None:
            print('[fallback build] mingw32-make / make not found – fallback unavailable')
            return

        omhome = os.environ.get('OPENMODELICAHOME', '')
        env = os.environ.copy()
        if omhome and platform.system() == 'Windows':
            extra = os.pathsep.join([
                os.path.join(omhome, 'bin'),
                os.path.join(omhome, 'tools', 'msys', 'ucrt64', 'bin'),
                os.path.join(omhome, 'lib', 'omc'),
            ])
            env['PATH'] = extra + os.pathsep + env.get('PATH', '')

        print(f'[fallback build] OMC build did not produce executable – running make manually for {model_name} ...')
        try:
            result = subprocess.run(
                [make_cmd, '-f', makefile],
                cwd=build_dir,
                env=env,
                timeout=600,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f'[fallback build] make failed (exit {result.returncode}):\n{result.stderr[-2000:]}')
                return
            if not os.path.exists(exe_path):
                print(f'[fallback build] make succeeded but {exe_name} not found in {build_dir}')
                return
            print(f'[fallback build] Successfully built {exe_name}')
        except subprocess.TimeoutExpired:
            print('[fallback build] make timed out after 600 s')
            return
        except FileNotFoundError:
            print(f'[fallback build] make executable not found: {make_cmd}')
            return
        except Exception as exc:
            print(f'[fallback build] Unexpected error: {exc}')
            return

        # Repair xmlFile and re-parse now that the exe (and init.xml) exist
        xml_candidate = os.path.join(build_dir, f'{model_name}_init.xml').replace('\\', '/')
        if os.path.exists(xml_candidate):
            self.xmlFile = xml_candidate
            self.xmlparse()
        else:
            print(f'[fallback build] Init XML not found: {xml_candidate}')

    def getSolutions(self, varList=None, resultfile=None):  # 12
        """
        This method returns tuple of numpy arrays. It can be called:
            •with a list of quantities name in string format as argument: it returns the simulation results of the corresponding names in the same order. Here it supports Python unpacking depending upon the number of variables assigned.
        usage:
        >>> getSolutions()
        >>> getSolutions("Name1")
        >>> getSolutions(["Name1","Name2"])
        >>> getSolutions(resultfile="c:/a.mat")
        >>> getSolutions("Name1",resultfile=""c:/a.mat"")
        >>> getSolutions(["Name1","Name2"],resultfile=""c:/a.mat"")
        """
        if resultfile == None:
            resFile = self.resultfile
        else:
            resFile = resultfile

        if not os.path.exists(resFile):
            errstr = "Error: Result file does not exist {}".format(resFile)
            self._raise_error(errstr=errstr)
            return
        else:
            resultVars = self.getconn.sendExpression("readSimulationResultVars(\"" + resFile + "\")")
            self.getconn.sendExpression("closeSimulationResultFile()")
            if varList is None:
                return resultVars
            elif isinstance(varList, str):
                if varList not in resultVars and varList != "time":
                    errstr = '!!! ' + varList + ' does not exist'
                    self._raise_error(errstr=errstr)
                    return None
                exp = "readSimulationResult(\"" + resFile + '",{' + varList + "})"
                res = self.getconn.sendExpression(exp)

                res = res.replace('{', '[').replace('}', ']')

                res = eval(res)

                exp2 = "closeSimulationResultFile()"
                self.getconn.sendExpression(exp2)
                return res
            elif isinstance(varList, list):
                for v in varList:
                    if v == "time":
                        continue
                    if v not in resultVars:
                        errstr = '!!! ' + v + ' does not exist'
                        self._raise_error(errstr=errstr)
                        return None
                variables = ",".join(varList)
                exp = "readSimulationResult(\"" + resFile + '",{' + variables + "})"
                res = self.getconn.sendExpression(exp, parsed=False)

                res = res.replace('{', '[').replace('}', ']')

                res = eval(res)

                exp2 = "closeSimulationResultFile()"
                self.getconn.sendExpression(exp2)
                return res
            else:
                return None

    def _raise_error(self, errstr):
        print(errstr)

    def setTempDirectory(self, customBuildDirectory):
        if customBuildDirectory is not None:
            os.makedirs(customBuildDirectory, exist_ok=True)
        super().setTempDirectory(customBuildDirectory)
