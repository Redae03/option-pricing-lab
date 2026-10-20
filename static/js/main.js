 (function () {
   let toastTimer = null;

   window.toast = function (msg) {
     const el = document.getElementById('toast');

     if (!el) {
       alert(msg);
       return;
     }

     el.textContent = msg;
     el.classList.add('show');

     clearTimeout(toastTimer);
     toastTimer = setTimeout(function () {
       el.classList.remove('show');
     }, 3000);
   };

   window.toggleDark = function () {
     document.body.classList.toggle('dark');

     try {
       const isDark = document.body.classList.contains('dark');
       localStorage.setItem('opl_dark_mode', isDark ? '1' : '0');
     } catch (e) {
       // localStorage indisponible
     }
   };

   document.addEventListener('DOMContentLoaded', function () {
     try {
       const savedDark = localStorage.getItem('opl_dark_mode');

       if (savedDark === '1') {
         document.body.classList.add('dark');
       }
     } catch (e) {
       // localStorage indisponible
     }
   });
 })();
