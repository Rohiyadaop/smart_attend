Option Explicit

Dim fso
Dim shell
Dim basePath
Dim launcherPath
Dim startupPath
Dim shortcutPath
Dim shortcut

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

basePath = fso.GetParentFolderName(WScript.ScriptFullName)
launcherPath = basePath & "\Start SmartAttend.vbs"
startupPath = shell.SpecialFolders("Startup")
shortcutPath = startupPath & "\SmartAttend.lnk"

If Not fso.FileExists(launcherPath) Then
  MsgBox "SmartAttend launcher file was not found.", vbCritical, "SmartAttend"
  WScript.Quit 1
End If

Set shortcut = shell.CreateShortcut(shortcutPath)
shortcut.TargetPath = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\wscript.exe")
shortcut.Arguments = """" & launcherPath & """"
shortcut.WorkingDirectory = basePath
shortcut.WindowStyle = 7
shortcut.Description = "Start SmartAttend automatically when Windows opens"
shortcut.IconLocation = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\SHELL32.dll,220")
shortcut.Save

MsgBox "SmartAttend will now start automatically when Windows opens.", vbInformation, "SmartAttend"
