Option Explicit

Dim fso
Dim shell
Dim shortcutPath

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

shortcutPath = shell.SpecialFolders("Startup") & "\SmartAttend.lnk"

If fso.FileExists(shortcutPath) Then
  fso.DeleteFile shortcutPath, True
  MsgBox "SmartAttend startup launch has been removed.", vbInformation, "SmartAttend"
Else
  MsgBox "SmartAttend startup shortcut was not found.", vbExclamation, "SmartAttend"
End If
