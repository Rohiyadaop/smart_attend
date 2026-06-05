Option Explicit

Dim fso
Dim shell
Dim basePath
Dim pythonwPath
Dim appPath
Dim command

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

basePath = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = basePath & "\venv\Scripts\pythonw.exe"
appPath = basePath & "\app.py"

If Not fso.FileExists(pythonwPath) Then
  MsgBox "SmartAttend could not find pythonw.exe in the virtual environment.", vbCritical, "SmartAttend"
  WScript.Quit 1
End If

If Not fso.FileExists(appPath) Then
  MsgBox "SmartAttend could not find app.py.", vbCritical, "SmartAttend"
  WScript.Quit 1
End If

shell.CurrentDirectory = basePath
command = """" & pythonwPath & """ """ & appPath & """"
shell.Run command, 0, False
